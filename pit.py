import Arena
from MCTS import MCTS

import numpy as np
from utils import *

import argparse

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

def main(args):

    human_vs_cpu = args.human_vs_cpu

    if args.game_name == "nim":
        from nim.nimGame import nimGame as Game
        from nim.nimPlayers import HumannimPlayer
        from nim.pytorch.NNet import NNetWrapper as NNet

        path = './pretrained_models/nim/pytorch/'

        g = Game(20,3)
        hp = HumannimPlayer(g).play
        args1 = dotdict({'numMCTSSims': 3, 'cpuct':1.0})

    elif args.game_name == "hex":
        from hex.hexGame import hexGame as Game
        from hex.hexPlayers import HumanhexPlayer
        from hex.pytorch.NNet import NNetWrapper as NNet

        path = './pretrained_models/hex/pytorch/'

        g = Game(5)
        hp = HumanhexPlayer(g).play
        args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})


    elif args.game_name == "pentago":
        from pentago.PentaGoGame import PentaGoGame as Game
        from pentago.PentaGoPlayers import HumanPentaGoPlayer
        from pentago.pytorch.NNet import NNetWrapper as NNet

        path = './pretrained_models/pentago/pytorch/'

        g = Game(6)
        hp = HumanPentaGoPlayer(g).play
        args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})


    else:
        raise NotImplementedError

    n1 = NNet(g)

    n1.load_checkpoint(path,'best.pth.tar')

    mcts1 = MCTS(g, n1, args1)

    ##change
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    ###

    if human_vs_cpu:
        player2 = hp
    else:
        n2 = NNet(g)
        n2.load_checkpoint(path, 'best.pth.tar')
        args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts2 = MCTS(g, n2, args2)
        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

        player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

    arena =  Arena.Arena(player2, n1p, g, display=Game.display)

    print(arena.playGames(2, verbose=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generating temperature models')

    parser.add_argument('--game_name', "--string",
                        default="pentago", type=str)
    parser.add_argument('--human_vs_cpu', default=True, action='store_true')
    parser.add_argument('--cpu_vs_cpu', dest='human_vs_cpu', action='store_false')
    main(parser.parse_args())