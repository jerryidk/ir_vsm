
'''represent each document as a vector'''
'''to compute, score = tf * idf'''
'''to compute, similarity = v_1 dot v_2'''

import numpy as np
from nltk.tokenize import word_tokenize
import buildindex

class VSM:

    def __init__(self, index: buildindex.Index, weight_func: str, numpy_file: str = None, b: int = 0.01):
        self.index = index
        print('Creating Word Index')
        self.word_index = self.createWordIndex()
        self.tf = np.zeros(0)

        print('Calculating Term Frequency')

        if numpy_file is not None:
            self.tfidf = np.load(numpy_file)
        else:
            print('No numpy file so will build term frequency')
            if weight_func == 'tf':
                print('Calculating tf')
                self.tf = self.calculateTermFrequency()
            elif weight_func == 'tflog':
                self.tf = self.calculateLogTermFrequency()
            elif weight_func == 'tfnorm':
                self.tf = self.calculateNormFrequency(b)
            else:
                self.tf = self.calculateTermFrequency()

            self.idf = self.calculateInverseDocumentFrequency()

            print('Calculating TF-IDF')

            self.tfidf = self.calculateTFIDF()

    def createWordIndex(self):
        """ associate word with position in vector """
        word_index = {}
        num  = 0
        for word in self.index.vocab:
            word_index[word] = num
            num += 1
        return word_index

    def calculateTermFrequency(self):
        """ get term frequency of all vocab terms """
        return np.array([[self.index.getTFinD(word, doc_id) for doc_id in self.index.doc_ids] for word in self.index.vocab]).T

    def calculateLogTermFrequency(self):
        """ get log term frequency of all vocab terms """
        return np.array([[1 + np.log(self.index.getTFinD(word, doc_id)) if self.index.getTFinD(word, doc_id) != 0
                        else 0 for doc_id in self.index.doc_ids] for word in self.index.vocab]).T

    def calculateNormFrequency(self, b: int = 0.01):
        """ get pivoted norm term frequency of all vocab terms """
        avgl = sum(self.index.doc_length.values())/len(self.index.doc_length)
        return np.array([[self.index.getTFinD(word, doc_id) / (1 - b + b * self.index.doc_length[doc_id]/avgl) for doc_id in self.index.doc_ids] for word in self.index.vocab]).T

    def calculateInverseDocumentFrequency(self):
        """ get inverse document frequency """
        return np.array([np.log(self.index.getDocNum() / self.index.getDocNumContainT(word)) for word in self.index.vocab])

    def calculateTFIDF(self):
        """ get tf-idf matrix """
        return np.multiply(self.tf, self.idf)

    def evaluateQuery(self, query : str, num_results : int):
        """ retrieve top n documents based on query """
        query_vector = np.zeros(len(self.index.vocab))

        tokens =  word_tokenize(query)
        query_token = tokens[1:]
        for word in query_token:
            #stem the word
            word = "".join(e for e in word if e.isalnum())
            word = self.index.stemmer.stem(word)
            if word in self.index.vocab:
                query_vector[self.word_index[word]] = np.log(self.index.getDocNum() / self.index.getDocNumContainT(word))

        scores = np.dot(query_vector, self.tfidf.T)

        # Sort scores
        scores_indices = np.argsort(scores)[::-1]

        results = {}

        for index in scores_indices[:num_results]:
            results[self.index.num_to_doc[index]] = scores[index]

        # Return top n results
        return results, scores
