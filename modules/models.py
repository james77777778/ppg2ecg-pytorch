import torch
from torch import nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        bs = x.size(0)
        return x.view(bs, -1)


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        if self.sigma != 0:
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class PPG2ECG(nn.Module):
    def __init__(self, input_size, use_stn=False, use_attention=False):
        super(PPG2ECG, self).__init__()
        self.use_stn = use_stn
        self.use_attention = use_attention
        # build main transformer
        self.main = nn.Sequential(
            # encoder
            nn.Conv1d(1, 32, kernel_size=31, stride=2, padding=15),
            nn.PReLU(32),
            nn.Conv1d(32, 64, 31, 1, 15),
            nn.PReLU(64),
            nn.Conv1d(64, 128, 31, 2, 15),
            nn.PReLU(128),
            nn.Conv1d(128, 256, 31, 1, 15),
            nn.PReLU(256),
            nn.Conv1d(256, 512, 31, 2, 15),
            nn.PReLU(512),
            # decoder
            nn.ConvTranspose1d(
                512, 256, kernel_size=31, stride=2,
                padding=15, output_padding=1),
            nn.PReLU(256),
            nn.ConvTranspose1d(256, 128, 31, 1, 15),
            nn.PReLU(128),
            nn.ConvTranspose1d(128, 64, 31, 2, 15, 1),
            nn.PReLU(64),
            nn.ConvTranspose1d(64, 32, 31, 1, 15),
            nn.PReLU(32),
            nn.ConvTranspose1d(32, 1, 31, 2, 15, 1),
            nn.Tanh(),
        )
        # build stn (optional)
        if use_stn:
            # pylint: disable=not-callable
            self.restriction = torch.tensor(
                [1, 0, 0, 0], dtype=torch.float, requires_grad=False)
            self.register_buffer('restriction_const', self.restriction)
            self.stn_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=1, out_channels=8, kernel_size=7, stride=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(
                    in_channels=8, out_channels=10, kernel_size=5, stride=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
            n_stn_conv = self.get_stn_conv_out(input_size)
            self.stn_fc = nn.Sequential(
                Flatten(),
                nn.Linear(n_stn_conv, 32),
                nn.ReLU(True),
                nn.Linear(32, 4)
            )
            self.stn_fc[3].weight.data.zero_()
            self.stn_fc[3].bias.data = torch.FloatTensor([1, 0, 1, 0])
        # build attention network (optional)
        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(),
                nn.Linear(input_size, input_size)
            )
            self.attn_len = input_size

    def get_stn_conv_out(self, input_size):
        bs = 1
        ch = 1
        x = torch.zeros([bs, ch, input_size])
        out_stn_conv = self.stn_conv(x)
        return out_stn_conv.data.view(bs, -1).size(1)

    def stn(self, x):
        xs = self.stn_conv(x)
        theta = self.stn_fc(xs)
        theta1 = theta[:, :2]
        theta2 = theta[:, 2:]
        theta1 = torch.cat(
            (self.restriction_const.repeat(theta1.size(0), 1), theta1), 1)
        theta1 = theta1.view(-1, 2, 3)
        # 1-d padding to 2-d for grid operations
        x = x.unsqueeze(-1)
        grid = F.affine_grid(theta1, x.size())
        x = F.grid_sample(x, grid, padding_mode='border')
        thetaw = theta2[:, 0].contiguous().view(x.size(0), 1, 1, 1)
        thetab = theta2[:, 1].contiguous().view(x.size(0), 1, 1, 1)
        x = torch.mul(x, thetaw)
        x = torch.add(x, thetab)
        # 2-d squeeze to 1-d
        x = x.squeeze(-1)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x1 = x
        # whether to use stn
        if self.use_stn:
            x2 = self.stn(x1)
        else:
            x2 = x
        # whether to use attention network
        if self.use_attention:
            attn_weights = F.softmax(self.attn(x2), dim=2)*self.attn_len
            x3 = x2*attn_weights
        else:
            x3 = x2
        # main transformer
        x4 = self.main(x3)
        return {'output': x4, 'output_stn': x2}


class PPG2ECG_BASELINE_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=200, num_layers=2):
        super(PPG2ECG_BASELINE_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = 0.1
        self.encoder = nn.LSTM(input_size, self.hidden_size, self.num_layers,
                               dropout=self.dropout, batch_first=True,
                               bidirectional=True)
        output_size = input_size
        self.decoder = nn.LSTM(self.hidden_size*2, output_size,
                               self.num_layers, batch_first=True,
                               bidirectional=True)
        self.linear = nn.Linear(output_size*2, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.input_size)
        encoded_output, (h_n, c_n) = self.encoder(x, None)
        encoded_output = nn.ReLU()(encoded_output)
        decoded_output, (h_n, c_n) = self.decoder(encoded_output, None)
        decoded_output = self.linear(decoded_output)
        decoded_output = decoded_output.view(x.size(0), -1, self.input_size)
        decoded_output = decoded_output.view(decoded_output.size(0), 1, -1)
        decoded_output = nn.Tanh()(decoded_output)
        return {'output': decoded_output}
