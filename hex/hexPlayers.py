import numpy as np
class HumanhexPlayer():
    def __init__(self, game):
        self.game = game
        self.n = game.getBoardSize()
        str_ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.str = str_
        self.str_trc = str_[:self.n[0]]

    def play(self, board):

        valid = self.game.getValidMoves(board, 1)

        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    y,x = [input_a[0], int(input_a[1])]

                    if not isinstance(y, str):
                        print("the first element has to be a string")
                    if (self.str_trc.find(y) >= 0 and (0 < x) and (x <= self.game.n)):
                        y_int = self.str_trc.find(y)
                        a = self.game.n * (x-1) + y_int
                        if valid[a]:
                            break
                except ValueError:
                    'Invalid integer'
            print('Invalid move')
        return a
    

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a
    

# class GreedyPlayer():
#     def __init__(self, game):
#         self.game = game

#     def play(self, board):
#         valids = self.game.getValidMoves(board, 1)
#         candidates = []
#         for a in range(self.game.getActionSize()):
#             if valids[a]==0:
#                 continue
#             nextBoard, _ = self.game.getNextState(board, 1, a)
#             score = self.game.getScore(nextBoard, 1)
#             candidates += [(-score, a)]
#         candidates.sort()
#         return candidates[0][1]