import sys
sys.path.append('..')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class hexNNet(nn.Module):
    def __init__(self, game, args):
        super(hexNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.conv1 = nn.Conv2d(1, args.num_channels, 3+2*args.pad_size-2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(args.num_channels, 1, 1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)
        self.bn5 = nn.BatchNorm2d(args.num_channels)
        self.bn6 = nn.BatchNorm2d(args.num_channels)
        self.bn7 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear((self.board_x)*(self.board_y), (self.board_x)*(self.board_y))
        self.fc_bn1 = nn.BatchNorm1d((self.board_x)*(self.board_y))

        self.fc2 = nn.Linear((self.board_x)*(self.board_y), (self.board_x)*(self.board_y))
        self.fc_bn2 = nn.BatchNorm1d((self.board_x)*(self.board_y))

        self.fc3 = nn.Linear((self.board_x)*(self.board_y), self.action_size)

        self.fc4 = nn.Linear((self.board_x)*(self.board_y), 1)

    def forward(self, s_, cote):
        pad = self.args.pad_size

        s = torch.zeros_like(s_)

        ##### Make the transpose of the matrix if board[1] is equal to -1 in order to make understandable this state for the neural network #####
        for i in range(cote.size(0)):
            if cote[i] == -1:
                s[i] = torch.transpose(s_[i],dim0=0,dim1=1)
            else:
                s[i] = s_[i]

    
        #### create a paddin of white and black of the board ###
        pad_w = (0,0,pad,pad)
        s = F.pad(s, pad_w, "constant", 1)

        pad_b = (pad,pad,0, 0)
        s = F.pad(s, pad_b, "constant", -1)

        s = s.view(-1, 1, self.board_x+2*pad, self.board_y+2*pad)                # batch_size x 1 x (board_x+2*pad) x (board_y+2*pad)
        s = F.relu(self.bn1(self.conv1(s))) # batch_size x num_channel x board_x x board_y
        s = F.dropout(s , p=self.args.dropout, training=self.training)
        s = F.relu(self.bn2(self.conv2(s))) # batch_size x num_channels x board_x x board_y
        s = F.dropout(s , p=self.args.dropout, training=self.training)
        s = F.relu(self.bn3(self.conv3(s))) # batch_size x num_channels x board_x x board_y
        s = F.dropout(s , p=self.args.dropout, training=self.training)
        s = F.relu(self.bn4(self.conv4(s))) # batch_size x num_channels x board_x x board_y
        s = F.dropout(s , p=self.args.dropout, training=self.training)
        s = F.relu(self.bn5(self.conv5(s))) # batch_size x num_channels x board_x x board_y
        s = F.dropout(s , p=self.args.dropout, training=self.training)
        s = F.relu(self.bn6(self.conv6(s))) # batch_size x num_channels x board_x x board_y
        s = F.dropout(s , p=self.args.dropout, training=self.training)
        s = F.relu(self.bn7(self.conv7(s))) # batch_size x num_channels x board_x x board_y
        s = F.dropout(s , p=self.args.dropout, training=self.training)

        s = s.view(-1, (self.board_x)*(self.board_y))

        pi = self.fc3(s) # batch_size x (board_x*board_y)
        v = self.fc4(s) # batch_size x 1

        ### If board[1]=-1 then convert the action to the corresponding transposed matrix

        pi_ = torch.zeros_like(pi)
        for k in range(pi.size(0)):
            if cote[k] == -1:
                work_matrix = torch.reshape(pi[k], (self.board_x, self.board_y))
                work_matrix = torch.transpose(work_matrix, dim0=0, dim1=1)
                pi_[k] = torch.reshape(work_matrix, ((self.board_x)* (self.board_y),))
            else:
                pi_[k] = pi[k]

        return pi_, torch.tanh(v)
