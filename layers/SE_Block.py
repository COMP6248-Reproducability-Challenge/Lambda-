
from collections import OrderedDict

from torch import nn

# class SELayer(nn.Module):
#     def __init__(self, Channel, reduction=16) -> None:
#         super(SELayer, self).__init__()
#         self.gobal_pooling = nn.AdaptiveAvgPool2d(1)
#         self.se = nn.Sequential(OrderedDict([
#           ('fc', nn.Linear(Channel, Channel // reduction, bias=False)),
#           ('relu', nn.ReLU(inplace=True)),# save memory
#           ('fc', nn.Linear(Channel // reduction, Channel, bias=False)),
#           ('sigmoid', nn.Sigmoid()),
#         ]))
        

#     def forward(self, x):
#         b, c, _, _ = x.size() # batch, channel
#         output = self.gobal_pooling(x).view(b,c)
#         output = self.se(output).view(b,c)
#         return x * output.expand_as(x)

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
