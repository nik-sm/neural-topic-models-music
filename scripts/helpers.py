import os
from time import gmtime, strftime

def makeExpDir():
    expPath = os.path.join("../results/experiments/",strftime("%Y_%m_%d_%H_%M_%S",gmtime()))
    os.mkdir(expPath)
    return expPath