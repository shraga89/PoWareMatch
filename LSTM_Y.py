import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class LSTM_Y(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size, device):
        super(LSTM_Y, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, input):
        # embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(input.view(len(input), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(input), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores
