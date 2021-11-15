
'''represent each document as a vector'''
'''to compute, score = tf * idf'''
'''to compute, similarity = v_1 dot v_2'''

import numpy as np
import buildindex

class VSM:

    def __init__(self, index: buildindex.Index, weight_func: str):
        self.index = index
        self.word_index = self.createWordIndex()
        self.tf = np.zeros(0)

        if weight_func == 'tf':
            self.tf = self.calculateTermFrequency()
        elif weight_func == 'tflog':
            self.tf = self.calculateLogTermFrequency()
        elif weight_func == 'tfnorm':
            self.tf = self.calculateTermFrequency()
        else:
            self.tf = self.calculateTermFrequency()

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
        tf = np.zeros((len(self.index.doc_ids), len(self.index.vocab)))
        for word in self.index.vocab:
            for doc_id in self.index.doc_ids:
                tf[self.index.doc_ids[doc_id]][self.word_index[word]] = self.index.getTFinD(word, doc_id)
        return tf

    def calculateLogTermFrequency(self):
        """ get log term frequency of all vocab terms """
        tf = np.zeros((len(self.index.doc_ids), len(self.index.vocab)))
        for word in self.index.vocab:
            for doc_id in self.index.doc_ids:
                if self.index.getTFinD(word, doc_id) != 0:
                    tf[self.index.doc_ids[doc_id]][self.word_index[word]] = 1 + np.log(self.index.getTFinD(word, doc_id))
                else:
                    tf[self.index.doc_ids[doc_id]][self.word_index[word]] = 0
        return tf

    def calculateDocumentFrequency(self):
        """ get document frequency """
        df = np.zeros(len(self.index.vocab))
        for word in self.index.vocab:
            df[self.word_index[word]] = self.index.getDocNumContainT(word)
        return df

    def calculateInverseDocumentFrequency(self):
        """ get inverse document frequency """
        df = self.calculateDocumentFrequency()
        idf = np.zeros(len(self.index.vocab))
        for word in self.index.vocab:
            idf[self.word_index[word]] = np.log(len(self.index.doc_ids) / df[self.word_index[word]])
        return idf

    def calculateTFIDF(self):
        """ get tf-idf matrix """
        tf = self.tf
        idf = self.calculateInverseDocumentFrequency()

        tfidf = np.zeros((len(self.index.doc_ids), len(self.index.vocab)))

        for doc_id in self.index.doc_ids:
            tfidf[self.index.doc_ids[doc_id]] = tf[self.index.doc_ids[doc_id]] * idf

        return tfidf

    def evaluateQuery(self, query: str, num_results: int):
        """ retrieve top n documents based on query """
        query_vector = np.zeros(len(self.index.vocab))

        for word in query.split():
            if word in self.index.vocab:
                query_vector[self.word_index[word]] = 1

        scores = np.dot(query_vector, self.tfidf.T)

        # Sort scores
        scores_indices = np.argsort(scores)[::-1]

        results = []

        for index in scores_indices[:num_results]:
            results.append(self.index.num_to_doc[index])

        # Return top n results
        return results
