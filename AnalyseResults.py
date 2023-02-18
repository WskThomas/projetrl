import Arena
from MCTS import MCTS
# from pentago.PentaGoGame import Game
# from pentago.PentaGoPlayers import *
# from pentago.pytorch.NNet import NNetWrapper as NNet

from hex.hexGame import hexGame as Game
from hex.hexPlayers import *
from hex.pytorch.NNet import NNetWrapper as NNet

import matplotlib.pyplot as plt


import numpy as np
from utils import *
import os

"""
use this script to test the performance of a model trained
"""

l_random_ratio = []
l_greedy_ratio = []
l_v0 = []
iter_max  = 35
arenaCompare = 20

g = Game(5)

rp = RandomPlayer(g).play
# gp = GreedyPlayer(g).play

n1 = NNet(g)
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})

for iter in range(1, iter_max+1):
    

    if os.path.exists('./temp/' + 'checkpoint_' + str(iter) + '.pth.tar'):

        n1.load_checkpoint('./temp/','checkpoint_' + str(iter) + '.pth.tar')
        print("ok")
    # _, v0 = n1.predict(np.zeros(6,6))
    _, v0 = n1.predict((np.zeros((5,5)),1))
    l_v0.append(v0)

    mcts1 = MCTS(g, n1, args)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


    arena_r = Arena.Arena(n1p, rp, g)
    # arena_g = Arena.Arena(n1p, gp, g)

    rpwins, rnwins, rdraws = arena_r.playGames(arenaCompare)
    # gpwins, gnwins, gdraws = arena_g.playGames(arenaCompare)

    l_random_ratio.append((rpwins, rnwins, rdraws))
    # l_greedy_ratio.append((gpwins, gnwins, gdraws))

l_iter = range(1, iter_max+1)


plt.plot(l_iter, l_v0, color = "black", label = "estimate of the value of an ititial board by the network")

plt.plot(l_iter, [rpwins/arenaCompare for rpwins, _, _ in l_random_ratio], color = "red", label = "ratio win against random agent")
# plt.plot(l_iter, [rdraws/arenaCompare for _, rdraws, _ in l_random_ratio], color = "yellow", label = "ratio draw against random agent")

# plt.plot(l_iter, [gpwins/arenaCompare for gpwins, _, _ in l_greedy_ratio], color = "blue", label = "ratio win against greedy agent")
# plt.plot(l_iter, [gdraws/arenaCompare for _, gdraws, _ in l_greedy_ratio], color = "green", label = "ratio draw against greedy agent")
plt.legend(bbox_to_anchor=(0.4, 0.8), loc="upper right")

plt.xlabel('iterations')

# plt.legend(loc = "upper left")
plt.title("Performance against a random agent")

plt.savefig("PerformanceBattleShip.png")
plt.show()




