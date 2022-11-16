import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from timm.models import rexnet

rex = rexnet.rexnet_200(pretrained=True)

rex.stem.conv = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
rex.head.fc = nn.Linear(in_features=2560, out_features=10, bias=True)

print(rex)

total_params = sum(p.numel() for p in rex.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(p.numel() for p in rex.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

from torch.autograd import Variable

def test_net():
    net = rex
    y = net(Variable((torch.randn(3,9,224,224))))
    print(y.size())
    
test_net()
