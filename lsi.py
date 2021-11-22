
import numpy as np
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import inv
import vsm

class LSI:

    def __init__(self, vsm: vsm.VSM):
        self.vsm = vsm
        self.lsi = self.calculateLSI()

    def calculateLSI(self, num_components: int = 100):
        # Run SVD
        self.U, self.S, self.VT = randomized_svd(self.vsm.tfidf.T, n_components=num_components,
                              n_iter='auto', random_state=None)

    def evaluateQuery(self, query : str, num_results : int):
        """ retrieve top n documents based on query """
        query_vector = np.zeros(len(self.vsm.index.vocab))

        for word in query.split():
            #stem the word
            word = self.vsm.stemmer.stem(word)
            if word in self.vsm.index.vocab:
                query_vector[self.vsm.word_index[word]] = 1

        # Calculate query vector
        query_vector = np.dot(np.dot(query_vector, self.U), inv(np.diag(self.S)))

        scores = np.dot(query_vector, self.VT)

        # Sort scores
        scores_indices = np.argsort(scores)[::-1]

        results = {}

        for index in scores_indices[:num_results]:
            results[self.vsm.index.num_to_doc[index]] = scores[index]

        # Return top n results
        return results