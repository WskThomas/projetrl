import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NimNNet(nn.Module):
    def __init__(self, game, args):
        super(NimNNet, self).__init__()
        self.game_size = game.getBoardSize()[0]
        self.action_size = game.getActionSize()
        self.args = args

        self.linear1 = nn.Linear(self.game_size, args.num_channel)
        self.linear2 = nn.Linear(args.num_channel, args.num_channel)
        # self.linear3 = nn.Linear(args.num_channel, args.num_channel)
        # self.linear4 = nn.Linear(args.num_channel, args.num_channel)
        # self.bn1 = nn.BatchNorm1d(args.num_channel)
        # self.bn2 = nn.BatchNorm1d(args.num_channel)
        # self.bn3 = nn.BatchNorm1d(args.num_channel)
        # self.bn4 = nn.BatchNorm1d(args.num_channel)


        self.linear1 = nn.Linear(args.num_channel, self.action_size)
        self.linear2 = nn.Linear(args.num_channel, 1)


        self.fc_bn1 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # x = self.bn1(self.linear1(x)) # batch_size x num_channel
        # x = F.relu(x)
        # x = F.dropout(x , p=self.args.dropout, training=self.training)

        # x = self.bn2(self.linear2(x)) # batch_size x num_channel
        # x = F.relu(x)
        # x = F.dropout(x , p=self.args.dropout, training=self.training)

        # x = self.bn3(self.linear3(x)) # batch_size x num_channel
        # x = F.relu(x)
        # x = F.dropout(x , p=self.args.dropout, training=self.training)

        # x = self.bn4(self.linear4(x)) # batch_size x num_channel
        # x = F.relu(x)
        # x = F.dropout(x , p=self.args.dropout, training=self.training)

        pi = self.linear1(x) # batch_size x action_size
        v = self.linear2(x) # batch_size x 1


        return pi, v