import torch.nn as nn
import torch
import math
from torch.autograd import Variable

import torch.nn.functional as F

def load_filmant(vocab, cfg):
    vocab_size = len(vocab['question_token_to_idx'])
    num_answers = len(vocab['answer_token_to_idx'])
    model = StepFilm(vocab_size, num_answers, cfg)
    if torch.cuda.is_available() and cfg.CUDA:
        model.cuda()
    else:
        model.cpu()
    model.train()
    return model


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
               transformer_dropout=0.1,
               transformer_heads=4,
               PE_dropout=0.1,
               n_operations = 1,
               ):
    super(QuestionToInstruction, self).__init__()
    self.n_instructions = n_instructions
    self.n_operations = n_operations
    encoderLayer = nn.TransformerEncoderLayer(d_model,transformer_heads, dropout=transformer_dropout)
    self.transformer = nn.TransformerEncoder(encoderLayer,
                                             transformer_nlayers)
    self.PE = PositionalEncoding(d_model, PE_dropout)

    self.encoder_embed = nn.Embedding(vocab_size + n_instructions + n_operations, d_model)
    # self.encoder_embed.weight.data.uniform_(-1, 1)
    self.embedding_dropout = nn.Dropout(p=0.15)

    self.instructions = torch.tensor([i for i in range(vocab_size, vocab_size + n_instructions)]).cuda()
    self.operations = torch.tensor([i for i in range(vocab_size + n_instructions,
                                                     vocab_size + n_instructions + n_operations)]).cuda()


  def forward(self, question, question_len):
    # # Delete the padding before passing to Transformer for efficiency
    # question = question[:question_len]    
    
    # Transform instruction and question to embedding and add PE to question
    embed_i = self.encoder_embed(self.instructions)
    embed_q = self.encoder_embed(question)
    embed_q = self.PE(embed_q)
    embed_q = self.embedding_dropout(embed_q)
    embed_o = self.encoder_embed(self.operations)
    
    # Transform instructions and operation tokens to match batch size. 
    embed_i = embed_i.unsqueeze(1).expand(self.n_instructions, question.shape[1], -1)
    embed_o = embed_o.unsqueeze(1).expand(self.n_operations, question.shape[1], -1)
    
    # Concat instruction tokens to questions (TODO: try difference between concat at the beginning or end)
#     print(embed_i.shape, embed_q.shape)
    embed = torch.cat((embed_i, embed_o, embed_q), 0)

    x = self.transformer(embed)
    return x[:self.n_instructions], x[self.n_instructions:self.n_instructions + self.n_operations]

class StepFilm(nn.Module):

  def __init__(self,
               vocab_size,
               num_answers,
               cfg,
               ):
    super(StepFilm, self).__init__()
    



    self.question_to_instruction = QuestionToInstruction(vocab_size,
                                                         d_model=cfg.model.d_model,
                                                         n_instructions=cfg.model.n_instructions,
                                                         transformer_nlayers=cfg.model.transformer_nlayers,
                                                         transformer_dropout=cfg.model.transformer_dropout,
                                                         transformer_heads=cfg.model.transformer_heads,
                                                         PE_dropout=cfg.model.PE_dropout,)
    self.img_input = nn.Sequential(nn.Dropout(p=0.2),
                                   nn.Conv2d(cfg.model.in_channels, cfg.model.cnn_dim, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Conv2d(cfg.model.cnn_dim, cfg.model.cnn_dim, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU())
    
    self.n_filmblocks = cfg.model.n_filmblocks
    self.cond_feat_size = 2 * cfg.model.cnn_dim
    self.cnn_dim = cfg.model.cnn_dim
    self.n_instructions = cfg.model.n_instructions
    self.n_operations = cfg.model.n_operations
    
    self.operation_linear = cfg.model.operation_linear
    if self.operation_linear:
      self.operationLinear = nn.Sequential(nn.Dropout(0.15),
                                      nn.Linear(cfg.model.d_model, cfg.model.d_model),
                                      nn.ReLU(),
                                      nn.Dropout(0.15),
                                      nn.Linear(cfg.model.d_model, cfg.model.d_model))

    # self.res_blocks = []
    # for _ in range(self.n_filmblocks):
    #         self.res_blocks.append(FiLMBlock(self.cnn_dim, dropout=0.2, batchnorm_affine=False))
    # self.res_blocks = nn.ModuleList(self.res_blocks)
    
    # self.film_generator = nn.Linear(cfg.model.d_model, self.cond_feat_size * self.n_filmblocks)
    self.instr_mem_join = nn.Linear(cfg.model.d_model*2, cfg.model.d_model)

    self.intruction_proj = nn.Linear(self.cnn_dim, cfg.model.d_model)
    self.activation = nn.ELU()
    self.dropout = nn.Dropout(0.15)
    self.attn = nn.Linear(cfg.model.d_model, 1)
    
    # Visualizations
    self.kb_attn_idty = nn.Identity()
    self.beta_idty = nn.Identity()
    self.gamma_idty = nn.Identity()
    self.res_block_idty = nn.Identity()
    self.features_idty = nn.Identity()
    
    # self.memory = nn.LSTM(cnn_dim, lstm_dim, 1)
    encoderLayer = nn.TransformerEncoderLayer(cfg.model.d_model,cfg.model.transformer_heads, dropout=cfg.model.transformer_dropout)
    self.transformer = nn.TransformerEncoder(encoderLayer,
                                             cfg.model.transformer_nlayers)

    self.PE = PositionalEncoding(cfg.model.d_model, cfg.model.PE_dropout)
    
    self.classifier = nn.Sequential(nn.Dropout(0.15),
                                    nn.Linear(cfg.model.d_model, cfg.model.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(0.15),
                                    nn.Linear(cfg.model.d_model, num_answers))
    
    
  def forward(self, question, question_len, image):
    question = question.permute(1,0)
    instructions, operations = self.question_to_instruction(question, question_len)
    if self.operation_linear:
      operations = self.operationLinear(operations)
    batch_size = instructions.shape[1]

    image = self.features_idty(image)
    img = self.img_input(image)
    img = img.view(batch_size, self.cnn_dim, -1)
    img = img.permute(0,2,1)
    mem = torch.empty(self.n_instructions + self.n_operations, batch_size, self.cnn_dim).cuda()

    for j, instruction in enumerate(instructions):
      # film = self.film_generator(instruction).view(batch_size, self.n_filmblocks,  self.cond_feat_size)
      # gammas, betas = torch.split(film[:,:,:2*self.cnn_dim], self.cnn_dim, dim=-1)
      # gammas = self.gamma_idty(gammas)
      # betas = self.beta_idty(betas)
      res = img
      # for i in range(len(self.res_blocks)):
      #   res = self.res_blocks[i](res, gammas[:, i, :], betas[:, i, :])
      # res = self.res_block_idty(res)
      #TODO: test max pool instead of sum
      # res = res.sum(1)
      # mem[j] = res
      #res = res.permute(0,2,1)
      # m = nn.MaxPool1d(res.shape[2])
      # res = m(res).permute(0,2,1)
      # mem[j] = res.squeeze(1)
      if j > 0:
        instruc = self.instr_mem_join(torch.cat((instruction, mem[j-1]), dim=1))
      else:
        instruc = instruction.clone()

      
      know = self.intruction_proj(res)
      instruc = instruc.unsqueeze(1)
      interactions = instruc * know
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
      mem[j] = read
    
    for j, op in enumerate(operations):
      mem[self.n_instructions + j] = op

    mem = self.PE(mem)
    x = self.transformer(mem)
    # x, _ = self.memory(mem, )
    #TODO change if n_operations >1
    x = x[-1]
    x = self.classifier(x)
    
    return x




import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import random

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient']) 

if __name__ == '__main__':
    model = StepFilm(40)
    #model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    #question = torch.randint(0,40,(20,32))
    question = torch.randint(0,40,(32,20))
    #question = question.cuda()
    image = torch.rand(32, 1024, 7, 7)
    #image = image.cuda()

    target = torch.zeros(32,28)#.cuda()
    for o in target:
        i = random.randint(0,27)
        o[i] = 1

    target = torch.max(target.long(), 1)[1]
    optimizer.zero_grad()
    output = model(question, 20, image)
    # loss = criterion(output, target)
    # #     print(loss)
    # loss.backward()
    # # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    # plot_grad_flow(model.named_parameters())
    # # loss.grad
    # # print(model.question_to_instruction.encoderLayer.self_attn.in_proj_weight.grad)
    # optimizer.step()
    # for name, param in model.named_parameters():
    # #     print(name)
    # #     print(name, param.grad.mean())
    #     if torch.isnan(param).any():
    #         print(name)
    #         break
