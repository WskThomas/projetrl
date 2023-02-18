# Project RL - AplaZero General
Implementation of different sets (nim, hex and pentago) based on the baseline you can find [here](https://github.com/suragnair/alpha-zero-general).
The provided baseline has been simplified (games such as tictactoe othello have been removed) in order to focus on the games that are required.

To use a game of your choice, subclass the classes in ```Game.py``` and ```NeuralNet.py``` and implement their functions. Example implementations for Othello can be found in ```hex/hexGame.py``` and ```hex/pytorch/NNet.py```. 

```Coach.py``` contains the core training loop and ```MCTS.py``` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in ```main.py```. Additional neural network parameters are in ```hex/pytorch/NNet.py``` (cuda flag, batch size, epochs, learning rate etc.). 

To start training a model for hex:
```bash
python main.py
```
Choose your framework and game in ```main.py```.

To play against a pretrained model:
```bash
python pit.py
```
Choose your framework and game in ```pit.py```.

