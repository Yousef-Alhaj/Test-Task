from main import torch,nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hx=None):
        if hx is None:
            h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        else:
            h0, c0 = hx
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
