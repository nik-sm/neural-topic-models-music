import lda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bow = pd.read_pickle("../data/nik/bag_of_words.pickle")

model = lda.LDA(n_topics=10, n_iter=100, random_state=1)

docTopicWeights = model.fit_transform(bow.values.astype(int))

np.save("../data/ldaFeatures.npy", docTopicWeights)


# # In[ ]:


# n_top_words = 15
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(topK.keys())[np.argsort(topic_dist)][:-n_top_words:-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# # In[ ]:


# doc_topic = model.doc_topic_


# # In[ ]:


# for i in range(10):
#     songNum = np.random.randint(0,topK.shape[0])
#     print('"{}" - {} (top topic: {})'.format(df["song"][songNum].replace("-"," "), df["artist"][songNum], doc_topic[songNum].argmax()))

