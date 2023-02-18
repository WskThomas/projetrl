from utils import *

import argparse
from MCTS import MCTS

from nim.nimGame import nimGame as Game
from nim.pytorch.NNet import NNetWrapper as NNet
import matplotlib.pyplot as plt
import numpy as np


path = './pretrained_models/nim/pytorch/'

g = Game(20,3)
nn= NNet(g)
nn.load_checkpoint(path,'best.pth.tar')

v = np.zeros((20))
pi = np.zeros((20,3))

for i in range(21):

    if i < 20:
        L = [0]*19
        for k in range(i):
            L[k] = 1
        input = (L,i)

        pi_, v_ = nn.predict(input)
        pi_ = np.flip(pi_)
        print("test", pi_)
        pi_ = np.exp(pi_)/sum(np.exp(pi_))
        g.display(input)
        pi[19-i] = pi_
        v[19-i] = v_

fig, (ax1,ax2) = plt.subplots(2,1)

import matplotlib.ticker as mticker

c = ax1.imshow(pi.T, extent=[0.5,20.5,0.5,3.5],cmap='binary')
ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax1.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax1.title.set_text("P(s,a)")
ax1.set_xlabel("s")
ax1.set_ylabel("a")
  

ax2.bar(x=np.arange(1,21),height=v)
ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax2.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax2.title.set_text("v")
ax2.set_xlabel("s")
# plt.colorbar(c)
ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))

plt.show()

    

