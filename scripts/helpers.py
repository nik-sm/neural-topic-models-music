import os
from time import gmtime, strftime
from gensim.models.callbacks import CallbackAny2Vec
import datetime

def makeExpDir():
    expPath = os.path.join("../results/experiments/",str(datetime.datetime.now()))
    expPath += "_1"
    while os.path.isfile(expPath):
        expPath = expPath[:-1]+expPath[-1]+1
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