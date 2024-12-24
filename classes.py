import torch
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super(MultiHeadAttention, self).__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    # d_model must be divisable by n_heads
    self.d_k = d_model // n_heads
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)

  def scaled_dot_product_attention(self, Q, K, V, mask = None):
    att_score = torch.matmul(Q, K.transpose(-2, -1) / math.sqrt(self.d_k))
    if mask is not None:
      att_score = att_score.masked_fill(mask == 0, -1e9)
    att_prob = torch.softmax(att_score, dim = -1)
    output = torch.matmul(att_prob, V)
    return output

  def split_heads(self, x):
    batch_size, seq_len, d_model = x.size()
    return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

  def combine_heads(self, x):
    batch_size, _, seq_len, d_k = x.size()
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

  def forward(self, Q, K, V, mask=None):
    Q = self.split_heads(self.w_q(Q))
    K = self.split_heads(self.w_k(K))
    V = self.split_heads(self.w_v(V))
    att_output = self.scaled_dot_product_attention(Q, K, V, mask)
    output = self.w_o(self.combine_heads(att_output))
    return output


class PositionWiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff):
    super(PositionWiseFeedForward, self).__init__()
    self.fc1 = nn.Linear(d_model, d_ff)
    self.fc2 = nn.Linear(d_ff, d_model)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len):
    super(PositionalEncoding, self).__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
  def __init__(self, d_model, n_heads, d_ff, dropout):
    super(Encoder, self).__init__()
    self.self_att = MultiHeadAttention(d_model, n_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
    self.norm = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    att_output = self.self_att(x, x, x, mask)
    x = self.norm(x + self.dropout(att_output))
    ff_output = self.feed_forward(x)
    x = self.norm(x + self.dropout(ff_output))
    return x

class Decoder(nn.Module):
  def __init__(self, d_model, n_heads, d_ff, dropout):
    super(Decoder, self).__init__()
    self.attention = MultiHeadAttention(d_model, n_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
    self.norm = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, encoder_output, que_mask, ans_mask):
    att_ouput = self.attention(x, x, x, ans_mask)
    x = self.norm(x + self.dropout(att_ouput))
    att_ouput = self.attention(x, encoder_output, encoder_output, que_mask)
    x = self.norm(x + self.dropout(att_ouput))
    ff_output = self.feed_forward(x)
    x = self.norm(x + self.dropout(ff_output))
    return x

class TorchTransformer(nn.Module):
  def __init__(self, ques_vocab_size, ans_vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, dropout):
    super(TorchTransformer, self).__init__()
    self.encoder_embedding = nn.Embedding(ques_vocab_size, d_model)
    self.decoder_embedding = nn.Embedding(ans_vocab_size, d_model)
    self.positional_encoding_ques = PositionalEncoding(d_model, max_seq_length)
    self.positional_encoding_ans = PositionalEncoding(d_model, max_seq_length)
    self.encoders = nn.ModuleList([Encoder(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
    self.decoders = nn.ModuleList([Decoder(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
    self.fc = nn.Linear(d_model, ans_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def generate_mask(self, ques, ans):
    ques_mask = (ques != 0).unsqueeze(1).unsqueeze(2)
    ans_mask = (ans != 0).unsqueeze(1).unsqueeze(3)
    seq_len = ans.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device= ans_mask.device), diagonal=1)).bool()
    ans_mask = ans_mask & nopeak_mask.to(device)
    return ques_mask.to(device), ans_mask.to(device)

  def forward(self, ques, ans):
    ques_mask, ans_mask = self.generate_mask(ques, ans)
    ques_embed = self.dropout(self.positional_encoding_ques(self.encoder_embedding(ques)))
    ans_embed = self.dropout(self.positional_encoding_ans(self.decoder_embedding(ans)))
    encoder_output = ques_embed
    for encoder in self.encoders:
      encoder_output = encoder(encoder_output, ques_mask)
    decoder_output = ans_embed
    for decoder in self.decoders:
      decoder_output = decoder(decoder_output, encoder_output, ques_mask, ans_mask)
    output = self.fc(decoder_output)
    return output
