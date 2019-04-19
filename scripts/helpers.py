import os
from time import gmtime, strftime

def makeExpDir():
    expPath = os.path.join("../results/experiments/",strftime("%Y_%m_%d_%H_%M_%S",gmtime()))
    os.mkdir(expPath)
    return expPath

class EpochLogger(CallbackAny2Vec):
'''Callback to log information about training'''

def __init__(self):
    self.epoch = 0

def on_epoch_begin(self, model):
    print("Epoch #{} start".format(self.epoch))

def on_epoch_end(self, model):
    print("Epoch #{} end".format(self.epoch))
    self.epoch += 1