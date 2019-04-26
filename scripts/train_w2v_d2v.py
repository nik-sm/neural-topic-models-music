import numpy as np
from helpers import *
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', action='store', help='Output directory', required=True)
    parser.add_argument('-i', '--infile', action='store', help='Input numpy file', required=True)
    parser.add_argument('-k', '--dimension', action='store', type=int, help='Vector dimension', required=True)
    args = parser.parse_args()


    print("Begin Word2Vec...")
    oldcorpus = np.load(args.infile, allow_pickle=True)
    #print("OLDCORPUS: ", oldcorpus)
    corpus=[]
    for l in oldcorpus:
        if l is not None and len(l) > 0:
            corpus.append(l)
        else:
            print("FOUND EMPTY SENTENCE") # TODO - why does the filtering upstream somehow leave empty sentences?
    #print("Length before: ", len(oldcorpus))
    #print("Length after: ", len(corpus))

    hParams = {
            "word2VecSize": args.dimension,
            "window":5,
            "min_count":2,
            "workers":4,
            "epochs": 100
    }

    model = Word2Vec(sentences=corpus, 
                     size=hParams["word2VecSize"],
                     window=hParams["window"],
                     min_count=hParams["min_count"],
                     workers=hParams["workers"],
                     callbacks=[EpochLogger()])

    model.train(corpus, total_examples=model.corpus_count, epochs=hParams["epochs"], callbacks=[EpochLogger()])
    model.save(os.path.join(args.outdir, "word2vec.model"))
    print("Finished Word2Vec")



    print("Begin Doc2Vec...")
    hParams = {
            "doc2VecSize" : args.dimension,
            "epochs" : 100,
            "min_count" : 2
    }

    corpus = [TaggedDocument(c, [i]) for i, c in enumerate(corpus)]

    model = Doc2Vec(vector_size=hParams["doc2VecSize"],
                    min_count=hParams["min_count"],
                    epochs=hParams["epochs"])
    model.build_vocab(corpus)

    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[EpochLogger()])
    model.save(os.path.join(args.outdir, "doc2vec.model"))
    print("Finished Doc2Vec")


if __name__ == "__main__":
    main()
