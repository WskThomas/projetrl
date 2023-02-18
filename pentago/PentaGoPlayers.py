import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanPentaGoPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)

        while True:

            print("subsquare, direction, square_x, square_y")
            input_move = input()
            input_a = input_move.split(" ")

            if len(input_a) == 4:
                try:
                    subsquare, direction, square_x, square_y = [int(i) for i in input_a]
                    if ((0 <= square_x) and (square_x < 6) and (0 <= square_y) and (square_y < 6) and (abs(direction) == 1) and (0 <= subsquare) and (subsquare < 4)):
                        direction = (direction + 1) // 2
                        a = 6*6*4*direction + subsquare * 6*6 + 6*square_x + square_y
        
                        if board[square_x][square_y] == 0:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')

        return a


class GreedyPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
