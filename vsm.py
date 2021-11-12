
'''represent each document as a vector'''
'''to compute, score = tf * idf'''
'''to compute, similarity = v_1 dot v_2'''

import numpy as np

class VSM:

    def __init__(self, index):
        self.vocab = self.createVocabList(index)
        self.doc_ids = self.createDocList(index)
        self.wordIndex = self.createWordIndex(self.vocab)
        self.docIndex = self.createDocIndex(self.doc_ids)
        self.tf = np.zeros(0)

    def createWordIndex(self, vocab):
        """ associate word with position in vector """
        wordIndex = {}
        index  = 0
        for word in vocab:
            wordIndex[word] = index
            index += 1
        return wordIndex

    def createDocIndex(self, doc_ids):
        """ associate document with position in vector """
        docIndex = {}
        index  = 0
        for doc_id in doc_ids.keys():
            self.docIndex[doc_id] = index
            index += 1
        return docIndex

    def createDocList(self, index):
        """ get dictionary of all documents """
        doc_ids = {}
        for word in index.keys():
            for doc_id in index[word].keys():
                doc_ids[doc_id] = 1
        return doc_ids

    def createVocabList(self, index):
        """ get dictionary of unique words in documents """
        vocab = {}
        for word in index.keys():
            vocab[word] = 1
        return vocab

    def calculateTermFrequency(self, index):
        """ get term frequency of all vocab terms """
        tf = np.zeros((len(self.doc_ids), len(self.vocab.keys())))
        for word in self.vocab.keys():
            for doc_id in self.doc_ids.keys():
                tf[self.docIndex[doc_id]][self.wordIndex[word]] = index[word][doc_id]
        return tf

    def calculateDocumentFrequency(self, index):
        """ get document frequency """
        df = np.zeros(len(self.vocab.keys()))
        for word in self.vocab.keys():
            frq = 0
            # For all documents which contain the word
            for doc_id in index[word].keys():
                frq = frq + 1
            df[self.wordIndex[word]] = frq
        return df

    def calculateInverseDocumentFrequency(self, index):
        """ get inverse document frequency """
        df = self.calculateDocumentFrequency(self, index)
        idf = np.zeros(len(self.vocab.keys()))
        for word in self.vocab:
            # Add 1 to prevent division by zero
            idf[self.wordIndex[word]] = np.log((len(self.doc_ids)) / (1 + df[self.wordIndex[word]]))
        return idf

    def calculateTFIDF(self, index):
        """ get tf-idf matrix """
        tf = self.calculateTermFrequency(index)
        idf = self.calculateInverseDocumentFrequency(index)

        tfidf = np.zeros((len(self.doc_ids), len(self.vocab.keys())))

        for doc_id in self.doc_ids.keys():
            tfidf[self.docIndex[doc_id]] = tf[self.docIndex[doc_id]] * idf

        return tfidf

    def evaluateQuery(self, query, tfidf, num_results):
        """ retrieve top n documents based on query """
        query_vector = np.zeros(len(query))

        for word in query:
            if word in self.vocab:
                query_vector[self.wordIndex[word]] = query.count(word)

        scores = np.dot(query_vector, tfidf.T)

        # Sort scores
        scores_indices = np.argsort(scores)

        # Return top n results
        return scores_indices[:num_results]
