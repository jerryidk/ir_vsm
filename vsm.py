
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
        return np.array([[self.index.getTFinD(word, doc_id) for doc_id in self.index.doc_ids] for word in self.index.vocab]).T

    def calculateLogTermFrequency(self):
        """ get log term frequency of all vocab terms """
        return np.array([[1 + np.log(self.index.getTFinD(word, doc_id)) if self.index.getTFinD(word, doc_id) != 0
                        else 0 for doc_id in self.index.doc_ids] for word in self.index.vocab]).T

    def calculateDocumentFrequency(self):
        """ get document frequency """
        return np.array([self.index.getDocNumContainT(word) for word in self.index.vocab])

    def calculateInverseDocumentFrequency(self):
        """ get inverse document frequency """
        df = self.calculateDocumentFrequency()
        idf = np.array([np.log(len(self.index.doc_ids) / df[self.word_index[word]]) for word in self.index.vocab])
        return idf

    def calculateTFIDF(self):
        """ get tf-idf matrix """
        tf = self.tf
        idf = self.calculateInverseDocumentFrequency()
        return tf * idf

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


    # Write query_id, delimitar Q0, doc_id, rank, rank_score, text galago