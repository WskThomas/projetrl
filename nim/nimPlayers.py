class HumannimPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        valid_str = ""
        for i in range(len(valid)):
            if valid[i]:
                if i < len(valid) - 1: valid_str += f"{i+1}, "
                else: valid_str += f"{i+1}."

        valid_str = "Choose the number of sticks to pull : " + valid_str
        print(valid_str)

        while True:
            input_move = input()
            if input_move not in ["1","2","3"]:
                print('Invalid move')
            else:
                input_move = int(input_move) - 1
                return input_move