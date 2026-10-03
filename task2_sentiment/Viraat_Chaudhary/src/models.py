"""RNN, LSTM and residual causal TCN classifiers; all weights learned from scratch.

Assistant-assisted implementation of the lineup accepted by Viraat.
RNN/LSTM API: https://docs.pytorch.org/docs/2.11/generated/torch.nn.LSTM.html
TCN concepts: Bai, Kolter & Koltun (2018), https://arxiv.org/abs/1803.01271
No teammate model code or pretrained weights are used.
"""
import torch
from torch import nn
from torch.nn import functional as F


def masked_pool(states, lengths):
    # states [batch, time, channels]. PAD positions never enter either pooling statistic.
    mask = torch.arange(states.shape[1], device=states.device)[None, :] < lengths[:, None]
    mean = (states * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1).to(states.dtype)
    maximum = states.masked_fill(~mask.unsqueeze(-1), torch.finfo(states.dtype).min).max(1).values
    return torch.cat((mean, maximum), dim=1)


class RecurrentClassifier(nn.Module):
    def __init__(self, config, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config['embedding_dim'], padding_idx=0)
        self.embedding_dropout = nn.Dropout(config['embedding_dropout'])
        kind = config['architecture']
        recurrent = nn.RNN if kind == 'rnn' else nn.LSTM
        options = {'nonlinearity': 'tanh'} if kind == 'rnn' else {}
        self.encoder = recurrent(config['embedding_dim'], config['hidden_dim'],
                                 num_layers=1, batch_first=True, bidirectional=False, **options)
        self.head = nn.Sequential(nn.LayerNorm(2 * config['hidden_dim']),
                                  nn.Linear(2 * config['hidden_dim'], config['head_dim']),
                                  nn.ReLU(), nn.Dropout(config['dropout']),
                                  nn.Linear(config['head_dim'], 1))

    def forward(self, ids, lengths):
        states, _ = self.encoder(self.embedding_dropout(self.embedding(ids)))
        # Right PAD tokens occur after valid states in this unidirectional encoder.
        return self.head(masked_pool(states, lengths)).squeeze(-1)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x, mask):
        z = self.conv1(F.pad(x, (self.left_padding, 0)))
        z = self.norm1(z.transpose(1, 2)).transpose(1, 2)
        z = self.dropout(F.relu(z)) * mask
        z = self.conv2(F.pad(z, (self.left_padding, 0)))
        z = self.norm2(z.transpose(1, 2)).transpose(1, 2)
        z = self.dropout(F.relu(z)) * mask
        return F.relu(x + z) * mask


class TCNClassifier(nn.Module):
    def __init__(self, config, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config['embedding_dim'], padding_idx=0)
        self.embedding_dropout = nn.Dropout(config['embedding_dropout'])
        self.projection = nn.Conv1d(config['embedding_dim'], config['channels'], 1)
        self.blocks = nn.ModuleList(CausalResidualBlock(config['channels'], config['kernel_size'],
                                                       d, config['dropout'])
                                    for d in config['dilations'])
        self.head = nn.Sequential(nn.LayerNorm(2 * config['channels']),
                                  nn.Linear(2 * config['channels'], config['head_dim']),
                                  nn.ReLU(), nn.Dropout(config['dropout']),
                                  nn.Linear(config['head_dim'], 1))
        self.receptive_field = 1 + 2 * (config['kernel_size'] - 1) * sum(config['dilations'])

    def forward(self, ids, lengths):
        mask = (torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None])[:, None, :]
        z = self.projection(self.embedding_dropout(self.embedding(ids)).transpose(1, 2)) * mask
        for block in self.blocks:
            z = block(z, mask)
        return self.head(masked_pool(z.transpose(1, 2), lengths)).squeeze(-1)


def build_model(config, vocabulary_size):
    if config['architecture'] in ('rnn', 'lstm'):
        return RecurrentClassifier(config, vocabulary_size)
    if config['architecture'] == 'tcn':
        return TCNClassifier(config, vocabulary_size)
    raise ValueError('Unsupported architecture: ' + config['architecture'])


def parameter_counts(model):
    return {'total_parameter_count': sum(p.numel() for p in model.parameters()),
            'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad)}
