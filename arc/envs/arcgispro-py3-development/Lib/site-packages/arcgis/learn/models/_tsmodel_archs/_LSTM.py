import torch
import torch.nn as nn


class _TSLSTM(nn.Module):
    def __init__(self, input_size=1, output_size=1, device=None, hidden_layer_size=100):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.device = device
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        outputs = []
        for batch in input_seq:
            batch = batch.squeeze()
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.device),
                                torch.zeros(1, 1, self.hidden_layer_size).to(self.device))

            lstm_out, self.hidden_cell = self.lstm(batch.view(len(batch), 1, -1).to(self.device), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(batch), -1))
            outputs += [predictions[-1]]

        outputs = torch.stack(outputs, 1)

        return outputs.to(self.device)