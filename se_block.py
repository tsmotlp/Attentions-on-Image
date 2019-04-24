from torch import nn
class SE_block(nn.Module):
    def __init__(self, num_features, reduction_factor=16):
        super(SE_block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(num_features, num_features // reduction_factor),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction_factor, num_features),
            nn.Sigmoid()
        )
    def forward(self, x):
        batch, channel, _, _ = x.size()
        squeeze_res = self.squeeze(x).view(batch, channel)
        excite_res = self.excite(squeeze_res)
        f_scale = excite_res.view(batch, channel, 1, 1)
        return x * f_scale