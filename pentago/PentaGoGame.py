from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .PentaGoLogic import Board
import numpy as np

class PentaGoGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }


    @staticmethod
    def getSquarePiece(piece):
        return PentaGoGame.square_content[piece]

    def __init__(self, n):
        self.n = n
  

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (6, 6)

    def getActionSize(self):
        # return number of actions
        # =board_size * board_size * subsquare * direction
        return 6 * 6 * 4 * 2

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move

        b = Board(self.n)
        b.pieces = np.copy(board)

        square = ((action % 36) // 6,  (action % 36) % 6)
        
        subsquare_x = ((action // 36) // 2) %2
        subsquare_y = (action // 36) % 2
        subsquare = subsquare_x, subsquare_y

        direction = action // (36*4)
        direction = 2 * direction - 1   #from 0 - 1 to -1 - 1

        move = (square, subsquare, direction)

        b.execute_move(move, player)

        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()

        b = Board(self.n)
        b.pieces = np.copy(board)

        for x in range(6):
            for y in range(6):
                if b[x][y] == 0:
                    for offset in range(8):
                        valids[6*x + y + 36 * offset] = 1
   
        return np.array(valids)

    def getGameEnded(self, board, player):

        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost, 1e-4 if draw
        # player = 1

        

        b = Board(self.n)
        b.pieces = np.copy(board)

        result = player * b.get_win_state()

        if abs(result) < 1:
            result = abs(result)

        return result


    def getScore(self, board, player):

        b = Board(self.n)
        b.pieces = np.copy(board)

        players_max = []

        for player in [-1, 1]:

            player_max = 0
            player_pieces = (b.pieces == player)

            for win_length in range(1, 6):

                if (b._is_straight_winner(player_pieces, win_length = win_length) or
                        b._is_straight_winner(player_pieces.transpose(), win_length = win_length) or
                        b._is_diagonal_winner(player_pieces, win_length = win_length)):

                    player_max = win_length

            players_max.append(player_max)


        if player == 1:
            return players_max[1] - players_max[0]

        else:
            return players_max[0] - players_max[1]





    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):

        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s



    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(PentaGoGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
