'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

import numpy as np
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    #__directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n):
        "Set up initial board configuration."
        self.n = n
        # Create the empty board array.
        self.pieces = np.zeros((6, 6), dtype='int')
        self.win_length = 5
      

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]


    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """

        moves = []    # stores the legal moves.

        #get all the available cases
        for x in range(6):
            for y in range(6):
                if self[x][y] == 0:
                    newmoves = self.get_moves_for_square((x,y))
                    moves +=  newmoves

        return moves

    def has_legal_moves(self, color):
        for y in range(6):
            for x in range(6):
                if self[x][y]==0:
                    return True
             
        return False

    def get_moves_for_square(self, square):
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        """

        (x,y) = square

        moves = []

        for subsquare_x in range(2):
            for subsquare_y in range(2):
                for direction in [-1, 1]:
                    subsquare = subsquare_x, subsquare_y
                    moves.append((square, subsquare, direction))

        return moves




    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        square, subsquare, direction = move
        (x, y) = square
        subsquare_x, subsquare_y = subsquare

        self[x][y] = color

        subsquare_proj = np.array(self[3 * subsquare_x : 3 * (subsquare_x+1), 3 * subsquare_y : 3 * (subsquare_y+1)]).copy()
        rot_subsquare_proj = np.rot90(subsquare_proj, k = direction)

        for i in range(3):
            for j in range(3):
                self[3*subsquare_x + i][3*subsquare_y + j] = rot_subsquare_proj[i][j]



    def get_win_state(self):

        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost, 1e-4 if draw
        # player = 1

        player_win = []

        for player in [-1, 1]:
            player_pieces = (self.pieces == player)
            # Check rows & columns for win
            if (self._is_straight_winner(player_pieces, win_length = self.win_length) or
                self._is_straight_winner(player_pieces.transpose(), win_length = self.win_length) or
                self._is_diagonal_winner(player_pieces, win_length = self.win_length)):
                
                player_win.append(player)

        if len(player_win) == 1:
            return player_win[0] 

        elif len(player_win) == 2:
            return 1e-4

        elif len(player_win) == 0:
            if self.has_legal_moves(1):
                return 0
            else:
                return 1e-4


    def _is_diagonal_winner(self, player_pieces, win_length):
         
        """Checks if player_pieces contains a diagonal win."""

        for i in range(len(player_pieces) - win_length + 1):
            for j in range(len(player_pieces[0]) - win_length + 1):
                if all(player_pieces[i + x][j + x] for x in range(win_length)):
                    return True
            for j in range(win_length - 1, len(player_pieces[0])):
                if all(player_pieces[i + x][j - x] for x in range(win_length)):
                    return True
        return False

    def _is_straight_winner(self, player_pieces, win_length):
        """Checks if player_pieces contains a vertical or horizontal win."""
        run_lengths = [player_pieces[:, i:i + win_length].sum(axis=1)
                       for i in range(len(player_pieces) - win_length + 2)]
        return max([x.max() for x in run_lengths]) >= win_length

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)): 
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])

