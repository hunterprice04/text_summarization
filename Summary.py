import numpy as np
import pandas as pd
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

"""
This logic here is taken from https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/\

In this file you will find the implementation of the text rank algorithm. The goals is to
take in an article or any amount of text and develop a summary of the text.
The algorithm is split into 6 steps:
    1. Concatenate all the text contained in the articles ****** take this out once impl. works
    2. Split the text into sentences
    3. Find the vector representation (word embeddings) for every sentence
    4. Calculate the similarities between the sentence vectors and store
        them into a similarity matrix.
    5. A certain number of top-ranked sentences form the final summary
"""


class Summary:
    def __init__(self, sentences, summary_size=5):
        """
        constructor parameters
        text            : list of sentences ex: ['this is sentence 1.', 'this is sentence 2']
        summary_size    : the number of sentences in the summary
        """

        # get necessary imports
        handleDownloads()
        from nltk.corpus import stopwords

        # get a list of english stop words
        # stopwords are commonly used words of a language (we want to get rid of these words)
        self.stopwords = stopwords.words('english')

        # clean the data of unnecessary values
        clean_sentences = self.cleanSentences(sentences)

        # Extract word vectors and create a dictionary with key value pairs 'word', [word vector]
        # https://nlp.stanford.edu/projects/glove/ for the precalculated word vectors
        self.word_embeddings = {}
        f = open('glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.word_embeddings[word] = coefs
        f.close()

        # create vectors for each of our sentences
        self.sentence_vectors = []
        self.createWordVectors(clean_sentences)

        # create a similarity matrix of the sentences
        self.sim_mat = np.zeros([len(sentences), len(sentences)])
        self.createSimilarityMatrix(len(sentences))

        # create a directed graph from the similarity matrix
        # then generate rankings for the sentences
        nx_graph = nx.from_numpy_array(self.sim_mat)
        scores = nx.pagerank(nx_graph)

        # extract top n sentences as the summary
        self.ranked_sentences = sorted(((scores[i], s)
                                        for i, s in enumerate(sentences)), reverse=True)
        self.ranked_sentences = self.ranked_sentences[:][1]
        self.summary = sentToText(self.ranked_sentences[:summary_size])



    # this takes a list of sentences and returns the cleaned sentences for the alg.
    def cleanSentences(self, sentences):
        # use regex expression to eliminate all non letter characters
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

        # make alphabets lowercase
        clean_sentences = [s.lower() for s in clean_sentences]

        # eliminate all stop words from the sentences
        clean_sentences = [removeStopwords(sent.split(), self.stopwords) for sent in clean_sentences]
        return clean_sentences

    # this will return a list of sentence vectors from the given clean sentences
    def createWordVectors(self, clean_sentences):
        # for each sentence we fetch vectors for their respective words
        # then we take the mean of those vectors to get a consolidated vector for the sentence
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([self.word_embeddings.get(w, np.zeros((100,)))
                         for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((100,))
            self.sentence_vectors.append(v)

    # this will create a similarity matrix of all of the sentences
    # the nodes of the matrix represent the sentences
    # the edges represent the similarity between the two sentences
    def createSimilarityMatrix(self, n):
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.sim_mat[i][j] = \
                        cosine_similarity(self.sentence_vectors[i].reshape(1, 100),
                                          self.sentence_vectors[j].reshape(1, 100))[0, 0]


#################################################################
# HELPER FUNCTIONS

# converts a list of sentences into one string separated by a user value
def sentToText(text, separator=' '):
    sentence = ""
    for s in text:
        sentence += s + separator
    return sentence


# this downloads the stopwords package in the nltk module needed for this file
def handleDownloads():
    # if the nltk package has not been downloaded, download it
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading nltk stopwords")
        nltk.download('stopwords')


# this will remove all stopwords from a given sentence and return the new cleaned sentence
# the sentence must be a list of words: ["this", "is", "a", "sentence"]
def removeStopwords(sen, stopwords):
    new_sent = " ".join([i for i in sen if i not in stopwords])
    return new_sent
