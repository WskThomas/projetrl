from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np


class nimGame(Game):
    def __init__(self, len_board, n_action):
        self.n = len_board
        self.n_action = n_action

    def getInitBoard(self):
        return [0]*self.n, 0
    
    def getBoardSize(self):
        return (self.n,)

    def getActionSize(self):
        return self.n_action

    def getNextState(self, board, player, action):
        n = board[1] + action + 1

        b_list = [1] * n + [0] * (self.n - n)
        return ((b_list, board[1] + action + 1), -player)

    def getValidMoves(self, board, player):
        for difference in range(1,self.n_action+1):
            if self.n - board[1] == 1+difference:
                list = [1]*difference + [0]*(self.n_action-difference)
                return np.array(list)
        return np.array([1] * self.n_action)


    def getGameEnded(self, board, player):
        if self.n - board[1] == 1:
            return -1
        else:
            return 0
    
    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        str = ''
        for i in range(self.n):
            if i < board[1]:
                str += '1'
            else:
                str += '0'
        return str

    @staticmethod
    def display(board):
        print('Current board:')
        dis = ""
        for i in range(len(board[0]) - board[1]):
            dis += "|  "
        print("")
        print(dis)
        print("")
