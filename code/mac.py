import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from utils import *


acts = {
    'RELU': nn.ReLU,
    'ELU': nn.ELU,
    'TANH': nn.Tanh,
    'NONE': nn.Identity,
}


def load_MAC(cfg, vocab):
    kwargs = {
        'vocab': vocab,
        'num_answers': len(vocab['answer_token_to_idx'])
    }

    model = MACNetwork(cfg, **kwargs)
    model_ema = MACNetwork(cfg, **kwargs)
    for param in model_ema.parameters():
        param.requires_grad = False

    if torch.cuda.is_available() and cfg.CUDA:
        model.cuda()
        model_ema.cuda()
    else:
        model.cpu()
        model_ema.cpu()
    model.train()
    return model, model_ema

class ControlUnit(nn.Module):
    def __init__(self,
                 module_dim,
                 max_step=4,
                 separate_syntax_semantics=False,
                 control_feed_prev=True,
                 control_cont_activation='TANH'
                ):
        super().__init__()
        self.attn = nn.Linear(module_dim, 1)
        self.control_input = nn.Sequential(nn.Linear(module_dim, module_dim),
                                           nn.Tanh())
        if control_feed_prev:
            self.cont_control = nn.Linear(2 * module_dim, module_dim)
            self.cont_control_act = acts[control_cont_activation]()
        else:
            self.cont_control = None
            self.cont_control_act = None
        self.cw_attn_idty = nn.Identity()

        self.control_input_u = nn.ModuleList()
        for i in range(max_step):
            self.control_input_u.append(nn.Linear(module_dim, module_dim))

        self.module_dim = module_dim
        self.control_feed_prev = control_feed_prev
        self.separate_syntax_semantics = separate_syntax_semantics

    def mask(self, question_lengths, device):
        max_len = max(question_lengths)
        mask = torch.arange(max_len, device=device).expand(len(question_lengths), int(max_len)) < question_lengths.unsqueeze(1)
        mask = mask.float()
        ones = torch.ones_like(mask)
        mask = (ones - mask) * (1e-30)
        return mask

    @staticmethod
    def mask_by_length(x, lengths, device=None):
        lengths = torch.as_tensor(lengths, dtype=torch.float32, device=device)
        max_len = max(lengths)
        mask = torch.arange(max_len, device=device).expand(len(lengths), int(max_len)) < lengths.unsqueeze(1)
        mask = mask.float().unsqueeze(2)
        x_masked = x * mask + (1 - 1 / mask)

        return x_masked

    def forward(self, question, context, question_lengths, step, prev_control=None):
        """
        Args:
            question: external inputs to control unit (the question vector).
                [batchSize, ctrlDim]
            context: the representation of the words used to compute the attention.
                [batchSize, questionLength, ctrlDim]
            control: previous control state
            question_lengths: the length of each question.
                [batchSize]
            step: which step in the reasoning chain
        """
        if self.separate_syntax_semantics:
            syntactics, semantics = context
        else:
            syntactics = semantics = context

        # compute interactions with question words
        question = self.control_input(question)
        question = self.control_input_u[step](question)

        if self.control_feed_prev:
            newContControl = self.cont_control(torch.cat((prev_control, question), dim=1))
            newContControl = self.cont_control_act(newContControl)
        else:
            newContControl = question

        newContControl = torch.unsqueeze(newContControl, 1)
        interactions = newContControl * syntactics

        # compute attention distribution over words and summarize them accordingly
        logits = self.attn(interactions)

        logits = self.mask_by_length(logits, question_lengths, device=syntactics.device)
        attn = F.softmax(logits, 1)
        attn = self.cw_attn_idty(attn)

        # apply soft attention to current context words
        next_control = (attn * semantics).sum(1)

        return next_control, question

class FiLMBlock(nn.Module):
    def __init__(self, module_dim, dropout=5e-2, batchnorm_affine=False):
        super(FiLMBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_proj = nn.Conv2d(module_dim, module_dim, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(module_dim, module_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(module_dim, affine=batchnorm_affine)

    def forward(self, know, gamma, beta):
        # Pass know through convolutions
        batch_size, hw, module_dim = know.size()
        fmap_size = int(math.sqrt(hw))
        know = know.transpose(1,2).view(batch_size, module_dim, fmap_size, fmap_size)

        x = self.relu(self.input_proj(know))

        # Store mid-result for residual connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.apply_film(out, gamma, beta)
        out = self.relu(self.dropout(out))
        residual = out + x
        return residual.permute(0, 2, 3, 1).view(batch_size, hw, module_dim)

    def apply_film(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (x * gammas) + betas

class ReadUnit(nn.Module):
    def __init__(self, module_dim, num_blocks, film_from):
        super().__init__()

        self.concat = nn.Linear(module_dim * 2, module_dim)
        # self.concat_2 = nn.Linear(module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)
        self.dropout = nn.Dropout(0.15)
        self.kproj = nn.Linear(module_dim, module_dim)
        self.mproj = nn.Linear(module_dim, module_dim)

        self.activation = nn.ELU()
        self.module_dim = module_dim
        self.kb_attn_idty = nn.Identity()

        self.num_blocks = num_blocks
        self.cond_feat_size = 2 * module_dim
        self.res_blocks = []
        for _ in range(num_blocks):
            self.res_blocks.append(FiLMBlock(self.module_dim, dropout=0.18, batchnorm_affine=False))
        self.res_blocks = nn.ModuleList(self.res_blocks) 
        self.film_generator = nn.Linear(self.module_dim, self.cond_feat_size * self.num_blocks)
        self.film_from = film_from

    def forward(self, memory, know, control, question_i, instruction, memDpMask=None):
        """
        Args:
            memory: the cell's memory state
                [batchSize, memDim]

            know: representation of the knowledge base (image).
                [batchSize, kbSize (Height * Width), memDim]

            control: the cell's control state
                [batchSize, ctrlDim]

            memDpMask: variational dropout mask (if used)
                [batchSize, memDim]
        """
        ## Step 0: Apply FiLMBlocks
        batch_size, _ = control.size()
        if self.film_from == 'qi':
            film = self.film_generator(question_i).view(batch_size, self.num_blocks,  self.cond_feat_size)
        elif self.film_from == 'control':
            film = self.film_generator(control).view(batch_size, self.num_blocks,  self.cond_feat_size)
        elif self.film_from == 'transformer':
            film = self.film_generator(instruction).view(batch_size, self.num_blocks,  self.cond_feat_size)
        gammas, betas = torch.split(film[:,:,:2*self.module_dim], self.module_dim, dim=-1)
        for i in range(len(self.res_blocks)):
            know = self.res_blocks[i](know, gammas[:, i, :], betas[:, i, :])

        ## Step 1: knowledge base / memory interactions
        # compute interactions between knowledge base and memory
        know = self.dropout(know)
        if memDpMask is not None:
            if self.training:
                memory = applyVarDpMask(memory, memDpMask, 0.85)
        else:
            memory = self.dropout(memory)
        know_proj = self.kproj(know)
        memory_proj = self.mproj(memory)
        memory_proj = memory_proj.unsqueeze(1)
        interactions = know_proj * memory_proj

        # project memory interactions back to hidden dimension
        interactions = torch.cat([interactions, know_proj], -1)
        interactions = self.concat(interactions)
        interactions = self.activation(interactions)
        # interactions = self.concat_2(interactions)

        ## Step 2: compute interactions with control
        control = control.unsqueeze(1)
        interactions = interactions * control
        interactions = self.activation(interactions)

        ## Step 3: sum attentions up over the knowledge base
        # transform vectors to attention distribution
        interactions = self.dropout(interactions)
        attn = self.attn(interactions).squeeze(-1)
        attn = F.softmax(attn, 1)
        attn = self.kb_attn_idty(attn)

        # sum up the knowledge base according to the distribution
        attn = attn.unsqueeze(-1)
        read = (attn * know).sum(1)

        return read


class WriteUnit(nn.Module):
    def __init__(
        self,
        module_dim,
        rtom=True,
        self_attn=False,
        gate=False,
        gate_shared=False,
    ):
        super().__init__()
        self.rtom = rtom
        self.self_attn = self_attn
        self.gate = gate
        self.gate_shared = gate_shared
        if self.rtom is False:
            self.linear = nn.Linear(module_dim * 2, module_dim)
            if self_attn:
                self.linear = nn.Linear(module_dim * 3, module_dim)
                self.ctrl_attn_proj = nn.Linear(module_dim, module_dim)
                self.ctrl_attn_linear = nn.Linear(module_dim, 1)
            else:
                self.linear = nn.Linear(module_dim * 2, module_dim)
                self.ctrl_attn_proj = None
                self.ctrl_attn_linear = None
            if gate:
                if gate_shared:
                    dim_gate_out = 1
                else:
                    dim_gate_out = module_dim
                self.ctrl_gate_linear = nn.Linear(module_dim, dim_gate_out)
            else:
                sel.ctrl_gate_linear = None
        else:
            self.linear = None

        
    def forward(self, memory, info, control=None, prev_controls=None, prev_memories=None):
        if self.rtom:
            newMemory = info
        else:
            newMemory = torch.cat([memory, info], -1)

            if self.self_attn:
                control = self.ctrl_attn_proj(control)
                prev_controls = torch.cat(prev_controls, dim=1)
                interactions = prev_controls * control.unsqueeze(1)
                attn = self.ctrl_attn_linear(interactions).squeeze(-1)
                attn = F.softmax(attn, 1).unsqueeze(-1)
                prev_memories = torch.cat(prev_memories, dim=1)
                self_smry = (attn * prev_memories).sum(1)
                
                newMemory = torch.cat([newMemory, self_smry], dim=-1)
            
            newMemory = self.linear(newMemory)

            if self.gate:
                control = self.ctrl_gate_linear(control)
                z = F.sigmoid(control)
                newMemory = newMemory * z + memory * (1 - z)

        return newMemory


class MACUnit(nn.Module):
    def __init__(self, units_cfg, module_dim=512, max_step=4):
        super().__init__()
        self.cfg = cfg
        self.control = ControlUnit(
            **{
                'module_dim': module_dim,
                'max_step': max_step,
                **units_cfg.common,
                **units_cfg.control_unit
            })
        self.read = ReadUnit(
            **{
                'module_dim': module_dim,
                **units_cfg.common,
                **units_cfg.read_unit,
            })
        self.write = WriteUnit(
            **{
                'module_dim': module_dim,
                **units_cfg.common,
                **units_cfg.write_unit,
            })

        self.initial_memory = nn.Parameter(torch.FloatTensor(1, module_dim))
        if cfg.model.init_mem == 'random':
            self.initial_memory.data.normal_()
        else:
            self.initial_memory.data.zero_()

        self.module_dim = module_dim
        self.max_step = max_step

    def zero_state(self, batch_size, question):
        initial_memory = self.initial_memory.expand(batch_size, self.module_dim)
        initial_control = question

        if self.cfg.TRAIN.VAR_DROPOUT:
            memDpMask = generateVarDpMask((batch_size, self.module_dim), 0.85)
        else:
            memDpMask = None

        return initial_control, initial_memory, memDpMask

    def forward(self, context, question, knowledge, question_lengths, instructions):
        batch_size = question.size(0)
        control, memory, memDpMask = self.zero_state(batch_size, question)
        controls = [control.unsqueeze(1)]
        memories = [memory.unsqueeze(1)]

        for i in range(self.max_step):
            # control unit
            control, question_i = self.control(question, context, question_lengths, i, prev_control=control)
            # read unit
            info = self.read(memory, knowledge, control, question_i, instructions[i], memDpMask)
            # write unit
            memory = self.write(memory, info, control,
                    prev_controls=controls, prev_memories=memories,
                )

            # For write self attn
            controls.append(control.unsqueeze(1))
            memories.append(memory.unsqueeze(1))

        return memory

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class QuestionToInstruction(nn.Module):
  def __init__(self,
               vocab_size,
               d_model=256,
               n_instructions=5,
               transformer_nlayers=6,
               transformer_heads=4,
               PE_dropout=0.1,
               ):
    super(QuestionToInstruction, self).__init__()
    self.n_instructions = n_instructions
    self.encoderLayer = nn.TransformerEncoderLayer(d_model,transformer_heads)
    self.transformer = nn.TransformerEncoder(self.encoderLayer,
                                             transformer_nlayers)
    self.PE = PositionalEncoding(d_model, PE_dropout)

    self.encoder_embed = nn.Embedding(vocab_size + n_instructions, d_model)
    # self.encoder_embed.weight.data.uniform_(-1, 1)
    self.embedding_dropout = nn.Dropout(p=0.15)
    self.instructions = torch.tensor([i for i in range(vocab_size, vocab_size + n_instructions)])

  def forward(self, question, question_len):
    # Delete the padding before passing to Transformer for efficiency
    # question = question[:question_len]    
    
    # Transform instruction and question to embedding and add PE to question
    embed_i = self.encoder_embed(self.instructions)
    embed_q = self.encoder_embed(question)
    embed_q = self.PE(embed_q)
    embed_q = self.embedding_dropout(embed_q)
    
    # Transform instruction tokens to match batch size. 
    embed_i = embed_i.unsqueeze(1).expand(self.n_instructions, question.shape[1], -1)
    
    # Concat instruction tokens to questions (TODO: try difference between concat at the beginning or end)
    embed = torch.cat((embed_i, embed_q), 0)

    x = self.transformer(embed)
    return x[:self.n_instructions], x[self.n_instructions:]


class InputUnit(nn.Module):
    def __init__(self,
                 vocab_size,
                 wordvec_dim=300,
                 rnn_dim=512,
                 module_dim=512,
                 bidirectional=True,
                 separate_syntax_semantics=False,
                 separate_syntax_semantics_embeddings=False,
                 stem_act='ELU',
                 in_channels=1024,
                ):
        super(InputUnit, self).__init__()

        self.dim = module_dim
        self.wordvec_dim = wordvec_dim
        self.separate_syntax_semantics = separate_syntax_semantics
        self.separate_syntax_semantics_embeddings = separate_syntax_semantics and separate_syntax_semantics_embeddings

        stem_act = acts[stem_act]
        self.stem = nn.Sequential(nn.Dropout(p=0.18),
                                  nn.Conv2d(in_channels, module_dim, 3, 1, 1),
                                  stem_act(),
                                  nn.Dropout(p=0.18),
                                  nn.Conv2d(module_dim, module_dim, kernel_size=3, stride=1, padding=1),
                                  stem_act())

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        if self.separate_syntax_semantics_embeddings:
            wordvec_dim *= 2
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.encoder_embed.weight.data.uniform_(-1, 1)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.08)

        self.transformer = QuestionToInstruction(vocab_size, wordvec_dim, 12)

    def forward(self, image, question, question_len):
        b_size = question.size(0)

        # get image features
        img = self.stem(image)
        img = img.view(b_size, self.dim, -1)
        img = img.permute(0,2,1)

        question_embedding, instructions = self.transformer(question, question_len)
        question_embedding = question_embedding.permute(1,0,2)
        # # get question and contextual word embeddings
        # embed = self.encoder_embed(question)
        # embed = self.embedding_dropout(embed)
        # if self.separate_syntax_semantics_embeddings:
        #     semantics = embed[:, :, self.wordvec_dim:]
        #     embed = embed[:, :, :self.wordvec_dim]
        # else:
        #     semantics = embed
        
        # embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True)
        # contextual_words, (question_embedding, _) = self.encoder(embed)
        
        # if self.bidirectional:
        #     question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        # question_embedding = self.question_dropout(question_embedding)

        # contextual_words, _ = nn.utils.rnn.pad_packed_sequence(contextual_words, batch_first=True)
        
        # if self.separate_syntax_semantics:
        #     contextual_words = (contextual_words, semantics)
        
        return question_embedding, contextual_words, img, instructions


class OutputUnit(nn.Module):
    def __init__(self, module_dim=512, num_answers=28):
        super(OutputUnit, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, memory):
        # apply classifier to output of MacCell and the question
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([memory, question_embedding], 1)
        out = self.classifier(out)

        return out


class MACNetwork(nn.Module):
    def __init__(self, cfg, vocab, num_answers=28):
        super().__init__()

        self.cfg = cfg
        if getattr(cfg.model, 'separate_syntax_semantics') is True:
            cfg.model.input_unit.separate_syntax_semantics = True
            cfg.model.control_unit.separate_syntax_semantics = True
            
        
        encoder_vocab_size = len(vocab['question_token_to_idx'])
        
        self.input_unit = InputUnit(
            vocab_size=encoder_vocab_size,
            **cfg.model.common,
            **cfg.model.input_unit,
        )

        self.output_unit = OutputUnit(
            num_answers=num_answers,
            **cfg.model.common,
            **cfg.model.output_unit,
        )

        self.mac = MACUnit(
            cfg.model,
            max_step=cfg.model.max_step,
            **cfg.model.common,
        )

        init_modules(self.modules(), w_init=cfg.TRAIN.WEIGHT_INIT)
        nn.init.uniform_(self.input_unit.encoder_embed.weight, -1.0, 1.0)
        nn.init.normal_(self.mac.initial_memory)

    def forward(self, image, question, question_len):
        # get image, word, and sentence embeddings
        question_embedding, contextual_words, img, instructions = self.input_unit(image, question, question_len)

        # apply MacCell
        memory = self.mac(contextual_words, question_embedding, img, question_len, instructions)

        # get classification
        out = self.output_unit(question_embedding, memory)

        return out

