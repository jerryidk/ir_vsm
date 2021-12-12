
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import inv
import vsm

class LSI:

    def __init__(self, vsm: vsm.VSM, topic: int, use_tf: bool = False):
        self.vsm = vsm
        if use_tf:
            self.lsi = self.calculateLSIWithTF(topic)
        else:
            self.lsi = self.calculateLSIWithTFIDF(topic)

    def calculateLSIWithTF(self, num_components: int = 100):
        # Run SVD
        self.U, self.S, self.VT = randomized_svd(self.vsm.tf.T, n_components=num_components,
                              n_iter='auto', random_state=None)

    def calculateLSIWithTFIDF(self, num_components: int = 100):
        # Run SVD
        self.U, self.S, self.VT = randomized_svd(self.vsm.tfidf.T, n_components=num_components,
                              n_iter='auto', random_state=None)

    def evaluateQuery(self, query : str, num_results : int):
        """ retrieve top n documents based on query """
        query_vector = np.zeros(len(self.vsm.index.vocab))

        tokens =  word_tokenize(query)
        query_token = tokens[1:]
        for word in query_token:
            #stem the word
            word = "".join(e for e in word if e.isalnum())
            word = self.vsm.index.stemmer.stem(word)
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
        return results, scores