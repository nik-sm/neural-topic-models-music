#!/usr/bin/env python
import sys
import os
import argparse

import pandas as pd
import numpy as np
import tensorflow as tf
import itertools,time
from collections import OrderedDict
from copy import deepcopy
from time import time
import pickle
import sys, getopt
from tf_model import VAE

'''--------------Netowrk Architecture and settings---------------'''

def make_network(decoder_dim,layer1=100,layer2=100,num_topics=50,bs=200,eta=0.002):
    tf.reset_default_graph()
    network_architecture = \
        dict(n_hidden_recog_1=layer1, # 1st layer encoder neurons
             n_hidden_recog_2=layer2, # 2nd layer encoder neurons
             n_hidden_gener_1=decoder_dim, # 1st layer decoder neurons
             n_input=decoder_dim, # MNIST data input (img shape: 28*28)
             n_z=num_topics)  # dimensionality of latent space
    batch_size=bs
    learning_rate=eta
    return network_architecture,batch_size,learning_rate



'''--------------Methods--------------'''
def create_minibatch(data, batch_size):
    rng = np.random.RandomState(10)

    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]

def save_weights_and_topic_prop(vae, beta, data, batch_size):
    beta_outfile = "beta.pickle"
    pd.DataFrame(beta).to_pickle(os.path.join(outdir, beta_outfile))
    theta_outfile = "theta_needsoftmax.pickle"
    theta = vae.topic_prop(X=data, batch_size=batch_size)
    pd.DataFrame(theta).to_pickle(os.path.join(outdir, theta_outfile))

def train(network_architecture, minibatches, n_samples_tr, learning_rate=0.001,
          batch_size=200, training_epochs=100, display_step=5):
    tf.reset_default_graph()
    vae = VAE(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    writer = tf.summary.FileWriter('logs', tf.get_default_graph())
    emb=0
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples_tr / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = next(minibatches)
            # Fit training using batch data
            cost,emb = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples_tr * batch_size

            if np.isnan(avg_cost):
                print(epoch,i,np.sum(batch_xs,1).astype(np.int),batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                # return vae,emb
                sys.exit()

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost))
    return vae,emb

def print_top_words(beta, feature_names, n_top_words=10):
    print('---------------Printing the Topics------------------')
    for row in range(len(beta)):
        print(" ".join([feature_names[word_idx] for word_idx in beta[row].argsort()[:-n_top_words - 1: -1]]))
    print('---------------End of Topics------------------')

def print_perp(model, docs_te):
    cost=[]
    for doc in docs_te:
        doc = doc.astype('float32')
        n_d = np.sum(doc)
        c=model.test(doc)
        cost.append(c/n_d)
    print('The approximated perplexity is: ',(np.exp(np.mean(np.array(cost)))))

def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def main():
    global vae, outdir
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='i', required=True, help='Input bag-of-words file')
    parser.add_argument('-o', action='store', dest='o', required=True, help='Output directory')
    parser.add_argument('-f', action='store', dest='f', required=True, help='Hidden units first layer', type=int)
    parser.add_argument('-s', action='store', dest='s', required=True, help='Hidden units second layer', type=int)
    parser.add_argument('-k', action='store', dest='k', required=True, help='Number of topics', type=int)
    parser.add_argument('-b', action='store', dest='b', required=True, help='Batch size', type=int)
    parser.add_argument('-r', action='store', dest='r', required=True, help='Learning rate', type=float)
    parser.add_argument('-e', action='store', dest='e', required=True, help='Training epochs', type=int)
    parsed_args = parser.parse_args()
    print("Begin prodLDA with arguments: ", parsed_args)
    f=parsed_args.f
    s=parsed_args.s
    k=parsed_args.k
    b=parsed_args.b
    r=parsed_args.r
    e=parsed_args.e
    infile = parsed_args.i
    outdir = parsed_args.o

    '''-----------Data--------------'''
    all_data = pd.read_pickle(infile)
    no_index = all_data.drop(all_data.columns[0], axis=1) # Drop the song index for this unsupervised training
    vocab = np.array(no_index.columns)
    vocab_size = len(vocab)
    data_tr = data_te = no_index.values
    docs_tr = data_tr
    docs_te = data_te
    n_samples_tr = data_tr.shape[0]
    n_samples_te = data_te.shape[0]


#--------------print the data dimentions--------------------------
    print('Data Loaded')
    print('Dim Training Data',data_tr.shape)
    print('Dim Test Data',data_te.shape)
    '''-----------------------------'''

#    n_samples_tr = data_tr.shape[0]
#    n_samples_te = data_te.shape[0]
#    network_architecture = \
#        dict(n_hidden_recog_1=100, # 1st layer encoder neurons
#             n_hidden_recog_2=100, # 2nd layer encoder neurons
#             n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
#             n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
#             n_z=50)  # dimensionality of latent space
#

    minibatches = create_minibatch(docs_tr.astype('float32'), batch_size=b)
    network_architecture,batch_size,learning_rate=make_network(layer1=f,
                                                               layer2=s,
                                                               num_topics=k,
                                                               bs=b,
                                                               eta=r,
                                                               decoder_dim=data_tr.shape[1])

    print(network_architecture)
    vae,beta = train(network_architecture=network_architecture, 
                     minibatches=minibatches, 
                     n_samples_tr=n_samples_tr, 
                     training_epochs=e,
                     batch_size=batch_size,
                     learning_rate=learning_rate)

    #print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])

    print_top_words(beta, vocab)
    print_perp(vae, docs_te)
    save_weights_and_topic_prop(vae, beta, docs_tr, batch_size)

if __name__ == "__main__":
   main()
