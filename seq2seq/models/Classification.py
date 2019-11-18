import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Classification(nn.Module):
    def __init__(self, hidden_size, num_layer, num_class, bidirectional=False, dropout_p=0, use_attention=False):
        super(Classification, self).__init__()

        self.hidden_size = hidden_size * 2
        self.num_layer = num_layer
        self.num_class = num_class
        self.use_attention = use_attention
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layer,
                            bidirectional=bidirectional,
                            dropout=dropout_p,
                            batch_first=True)
        self.linear_1 = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size * self.num_directions * self.num_layer, self.hidden_size)

        input_size = self.hidden_size * self.num_directions + self.hidden_size
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 2 // 2)
        self.fc3 = nn.Linear(input_size // 2 // 2, self.num_class)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def attention_layer(self, lstm_output, final_state):
        """

        :param lstm_output: [batch_size, seq_len, hidden_size * num_directions]
        :param final_state: [num_layers * num_directions, batch_size, hidden_size]
        :return:
        """
        batch_size = lstm_output.size(0)
        seq_len = lstm_output.size(1)
        # [num_layers * num_directions, batch_size, hidden_size] -> [num_layers, batch_size, hidden_size * num_directions]
        hidden = torch.cat([final_state[0:final_state.size(0):2], final_state[1:final_state.size(0):2]], 2)
        # [batch_size, hidden_size * num_directions * num_layers]
        hidden = self.linear_2(hidden.transpose(0, 1).view(batch_size, -1))

        atten_lstm = torch.tanh(self.linear_1(lstm_output))
        atten_weights = torch.bmm(atten_lstm, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(atten_weights, 1)

        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state, soft_attn_weights

    def forward(self, x, decoder_hidden=None):
        output, (h_n, c_n) = self.lstm(x)
        # batch_size = output.size(0)
        # seq_len = output.size(1)
        # h = output.size(2)

        output, attn = self.attention_layer(output, h_n)
        output = torch.cat((output, decoder_hidden), dim=1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output, attn

