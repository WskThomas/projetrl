import numpy as np

class DisJointSets():
    def __init__(self,N):
        # Initially, all elements are single element subsets
        self._parents = [node for node in range(N)]
        self._ranks = [1 for _ in range(N)]

    def find(self, u):
        while u != self._parents[u]:
            # path compression technique
            self._parents[u] = self._parents[self._parents[u]]
            u = self._parents[u]
        return u

    def connected(self, u, v):
        return self.find(u) == self.find(v)

    def union(self, u, v):
        # Union by rank optimization
        root_u, root_v = self.find(u), self.find(v)
        if root_u == root_v:
            return True
        if self._ranks[root_u] > self._ranks[root_v]:
            self._parents[root_v] = root_u
        elif self._ranks[root_v] > self._ranks[root_u]:
            self._parents[root_u] = root_v
        else:
            self._parents[root_u] = root_v
            self._ranks[root_v] += 1
        return False



def isValid(np_shape: tuple, index: tuple):
    index = np.array(index)
    return (index >= 0).all() and (index < np_shape).all()

class Board():
    def __init__(self, n):
        "Set up initial board configuration."

        self.n = n
        self.b = np.zeros((self.n, self.n))

        self.current_player_structure = DisJointSets(n*n)
        self.other_player_structure = DisJointSets(n*n)


    def get_pieces(self):
        return self.b

    def add_uf_structure(self, board, uf_structure, action, player, n):
        i = action//n
        j = action%n
        possible_action = [[i-1,j], [i+1,j], [i, j-1], [i,j+1], [i+1,j-1], [i-1,j+1]]
        for action_ in possible_action:
            if not isValid((n,n), action_):
                continue
            action_list_format = n*action_[0]+action_[1]
            if board[action_[0],action_[1]] == player and not uf_structure.connected(action, action_list_format):
                uf_structure.union(action, action_list_format)

    def update(self, player, action):
        i = action//self.n
        j = action%self.n
        self.b[i, j] = player


    def valid_move(self):
        valid_move = np.zeros((self.n*self.n))
        for k in range(self.n*self.n):
            i = k//self.n
            j = k%self.n
            if self.b[i, j] == 0: valid_move[k] = 1
        return valid_move

    def check_won(self, player, cote):
        n = self.n
        b_work = self.b.copy()
        if cote==1:
            b_work = self.b.copy()
        if cote==-1:
            b_work =self.b.T.copy()

        uf_structure_player = DisJointSets(n*n)
        uf_structure_enemy = DisJointSets(n*n)
        for i in range(n):
            for j in range(n):
                if b_work[i,j] !=0:
                    k = i*n+j
                    possible_action = [[i-1,j], [i+1,j], [i, j-1], [i,j+1], [i+1,j-1], [i-1,j+1]]
                    for action in possible_action:
                        if not isValid((n,n), action):
                            continue
                        if b_work[action[0], action[1]] == b_work[i,j]:
                            action_list = n*action[0]+action[1]
                            if b_work[i,j] == player:
                                uf_structure_player.union(k,action_list)
                            elif b_work[i,j] == -player:            
                                uf_structure_enemy.union(k,action_list)

        board_1 = np.arange(0,n)
        board_2 = np.arange(n*(n-1),n*n)
        board_3 = np.arange(0,n*n,step=n)
        board_4 = np.arange(n-1,n*n,step=n)


        if player==1:
            for i in range(len(board_1)):
                board_1[i] = uf_structure_player.find(board_1[i])
                board_2[i] = uf_structure_player.find(board_2[i])
                board_3[i] = uf_structure_enemy.find(board_3[i])
                board_4[i] = uf_structure_enemy.find(board_4[i])

                for parent in board_1:
                    if parent in board_2: return 1

                for parents in board_3:
                    if parents in board_4: return -1

        if player==-1:
            for i in range(len(board_1)):
                board_1[i] = uf_structure_enemy.find(board_1[i])
                board_2[i] = uf_structure_enemy.find(board_2[i])
                board_3[i] = uf_structure_player.find(board_3[i])
                board_4[i] = uf_structure_player.find(board_4[i])

                for parent in board_3:
                    if parent in board_4: return 1

                for parents in board_1:
                    if parents in board_2: return -1

        
        for parent in board_1:
            if parent in board_2: return 1

        for parents in board_3:
            if parents in board_4: return -1

        return 0

        

    def canonical(self, player):
        self.b = self.b*player
        # if player == -1:
        #     self.b = self.b.T

    def tostring(self):
        return str(self.b)