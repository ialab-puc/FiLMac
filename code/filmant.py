import torch.nn as nn
import torch
import math
from torch.autograd import Variable

def load_filmant(vocab, cfg):
    vocab_size = len(vocab['question_token_to_idx'])
    num_answers = len(vocab['answer_token_to_idx'])
    model = StepFilm(vocab_size, num_answers=num_answers)
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
    self.instructions = torch.tensor([i for i in range(vocab_size, vocab_size + n_instructions)]).cuda()

  def forward(self, question, question_len):
    # Delete the padding before passing to Transformer for efficiency
    question = question[:question_len]    
    
    # Transform instruction and question to embedding and add PE to question
    embed_i = self.encoder_embed(self.instructions)
    embed_q = self.encoder_embed(question)
    embed_q = self.PE(embed_q)
    embed_q = self.embedding_dropout(embed_q)
    
    # Transform instruction tokens to match batch size. 
    embed_i = embed_i.unsqueeze(1).expand(self.n_instructions, question.shape[1], -1)
    
    # Concat instruction tokens to questions (TODO: try difference between concat at the beginning or end)
#     print(embed_i.shape, embed_q.shape)
    embed = torch.cat((embed_i, embed_q), 0)

    x = self.transformer(embed)
    return x[:self.n_instructions]

class StepFilm(nn.Module):

  def __init__(self,
               vocab_size,
               d_model=256,
               n_instructions=5,
               transformer_nlayers=6,
               transformer_heads=4,
               PE_dropout=0.1,
               n_filmblocks=3,
               in_channels = 1024,
               cnn_dim=512,
               lstm_dim=512,
               num_answers=28
               ):
    super(StepFilm, self).__init__()
    
    self.question_to_instruction = QuestionToInstruction(vocab_size,
                                                         d_model=256,
                                                         n_instructions=5,
                                                         transformer_nlayers=6,
                                                         transformer_heads=4,
                                                         PE_dropout=0.1,)
    self.img_input = nn.Sequential(nn.Dropout(p=0.2),
                                   nn.Conv2d(in_channels, cnn_dim, 3, 1, 1),
                                   nn.ReLU())
    
    self.n_filmblocks = n_filmblocks
    self.cond_feat_size = 2 * cnn_dim
    self.cnn_dim = cnn_dim
    self.n_instructions = n_instructions
    
    self.res_blocks = []
    for _ in range(n_filmblocks):
            self.res_blocks.append(FiLMBlock(self.cnn_dim, dropout=0.2, batchnorm_affine=False))
    self.res_blocks = nn.ModuleList(self.res_blocks)
    
    self.film_generator = nn.Linear(d_model, self.cond_feat_size * self.n_filmblocks)
    
    self.memory = nn.LSTM(cnn_dim, lstm_dim, 1)
    
    self.classifier = nn.Sequential(nn.Dropout(0.15),
                                    nn.Linear(lstm_dim, lstm_dim),
                                    nn.ReLU(),
                                    nn.Dropout(0.15),
                                    nn.Linear(lstm_dim, num_answers))
    
    
  def forward(self, question, question_len, image):
    instructions = self.question_to_instruction(question, question_len)
    batch_size = instructions.shape[1]
    img = self.img_input(image)
    img = img.view(batch_size, self.cnn_dim, -1)
    img = img.permute(0,2,1)
    mem = torch.empty(self.n_instructions, batch_size, self.cnn_dim).cuda()
    for i, instruction in enumerate(instructions):
      film = self.film_generator(instruction).view(batch_size, self.n_filmblocks,  self.cond_feat_size)
      gammas, betas = torch.split(film[:,:,:2*self.cnn_dim], self.cnn_dim, dim=-1)
      res = img
      for i in range(len(self.res_blocks)):
        res = self.res_blocks[i](res, gammas[:, i, :], betas[:, i, :])
      #TODO: test max pool instead of sum
      res = res.sum(1)
      mem[i] = res
    
    x, _ = self.memory(mem, )
    x = x[-1]
    x = self.classifier(x)
    
    return x