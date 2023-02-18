from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np
from .hexLogic import Board



class hexGame(Game):
    def __init__(self, len_board):
        self.n = len_board

    def getInitBoard(self):
        board = Board(self.n)
        return (board.b, 1)
    
    def getBoardSize(self):
        return (self.n,self.n)

    def getActionSize(self):
        return self.n*self.n

    def getNextState(self, board, player, action):
        new_board = Board(self.n)
        new_board.b = np.copy(board[0])
        new_board.update(player, action)
        cote = board[1]
        return ((new_board.b, cote), -player)

    def getValidMoves(self, board, player):
        new_board = Board(self.n)
        new_board.b = np.copy(board[0])
        return new_board.valid_move()


    def getGameEnded(self, board, player):
        new_board = Board(self.n)
        new_board.b = np.copy(board[0])
        cote = board[1]
        return new_board.check_won(player,cote=cote)
    
    def getCanonicalForm(self, board, player):
        return (board[0]*player, player*board[1])

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        if board[1] == 1:
            return str(board[0].astype(np.int))
        else:
            return str(-board[0].T.astype(np.int))
        # return str(board[0].astype(np.int))+str(board[1])

    @staticmethod
    
    def display(board_):
        print("")
        print("Rules recap:")
        print("The player 1 (W pawn) must connect the letters: the other player must connect the numbers.")
        print('Example of a move: B 2')
        print("")
        column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        board = board_[0].astype(np.int)
        if board_[1] == -1:
            board = board.T
        rows = len(board)
        cols = len(board[0])
        indent = 0
        headings = " "*5+(" "*3).join(column_names[:cols])
        # R = range(1,rows+1)
        # headings = " "*5+(" "*3).join([str(e) for e in R])
        print(headings)
        tops = " "*5+(" "*3).join("-"*cols)
        print(tops)
        roof = " "*4+"/ \\"+"_/ \\"*(cols-1)
        print(roof)
        color_mapping = lambda i : "B W"[i+1]
        for r in range(rows):
            row_mid = " "*indent
            row_mid += " {} | ".format(r+1)
            # row_mid += " {} | ".format(column_names[r])
            row_mid += " | ".join(map(color_mapping,board[r]))
            row_mid += " | {} ".format(r+1)
            # row_mid += " | {}".format(column_names[r])
            print(row_mid)
            row_bottom = " "*indent
            row_bottom += " "*3+" \\_/"*cols
            if r<rows-1:
                row_bottom += " \\"
            print(row_bottom)
            indent += 2
        headings = " "*(indent-2)+headings
        print(headings)