import logging

import coloredlogs

from Coach import Coach
import argparse
from nim.nimGame import nimGame as Game
from nim.pytorch.NNet import NNetWrapper as nn

from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.



def main(argpars):

    if argpars.game_name == "hex":
        from hex.hexGame import hexGame as Game
        from hex.pytorch.NNet import NNetWrapper as nn

        log.info('Loading %s...', Game.__name__)
        g = Game(5)
        args = dotdict({
            'numIters': 1000,
            'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
            'tempThreshold': 15,        #
            'updateThreshold': 0.51,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
            'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
            'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
            'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
            'cpuct': 1,

            'checkpoint': './temp/',
            'load_model': False,
            'load_folder_file': ('./temp/','checkpoint_80.pth.tar'),
            'numItersForTrainExamplesHistory': 20,
        })
    
    elif argpars.game_name == "nim":
        from nim.nimGame import nimGame as Game
        from nim.pytorch.NNet import NNetWrapper as nn

        log.info('Loading %s...', Game.__name__)
        g = Game(20,3)

        args = dotdict({
            'numIters': 1000,
            'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
            'tempThreshold': 20,        #
            'updateThreshold': 0.5,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
            'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
            'numMCTSSims': 5,          # Number of games moves for MCTS to simulate.
            'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
            'cpuct': 2,

            'checkpoint': './temp/',
            'load_model': False,
            'load_foldefile': ('/dev/models/8x100x50','best.pth.tar'),
            'numItersForTrainExamplesHistory': 20,
        })

    elif argpars.game_name == "pentago":
        from pentago.PentaGoGame import PentaGoGame as Game
        from pentago.pytorch.NNet import NNetWrapper as nn

        log.info('Loading %s...', Game.__name__)
        g = Game(6)
        args = dotdict({
            'numIters': 1000,
            'numEps': 100,       #100       # Number of complete self-play games to simulate during a new iteration.
            'tempThreshold': 15,        #
            'updateThreshold': 0.51,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
            'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
            'numMCTSSims': 50,     #100     # Number of games moves for MCTS to simulate.
            'arenaCompare': 80,         # Number of games to play during arena play to determine if new net will be accepted.
            'cpuct': 1,

            'checkpoint': './temp/',
            'load_model': False,
            'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
            'numItersForTrainExamplesHistory': 20,

        })


    else:
        raise NotImplementedError
    
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training script')

    parser.add_argument('--game_name', "--string",
                        default="hex", type=str)
    main(parser.parse_args())