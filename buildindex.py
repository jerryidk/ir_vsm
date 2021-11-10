from nltk.tokenize import word_tokenize
pathToCorpus = '/home/jerry/workspace/cs_6550/proj/nfcorpus/dev.docs'


class Index:

    '''index is a dict
       {word_1 : {doc_id_1 : tf, doc_id_2 : tf}
        word_2 : {doc_id_1 : tf}, ...} ,
         ....  : .... }
    '''
    def __init__(self):
        self.vocab = {}
        
    def processIndex(self, input, output):
        pass

    #done last
    def processLatentIndex(self, input, output):
        pass

    #return word tf in a given document 
    def getTFinD(self, word, docId):
        if(self.vocab.has_key(word) and self.vocab[word].has_key(docId)):
            return self.vocab[word][docId]
        else:
            print("word or docId doesn't exist in this index")

    #return total documents in corpus containing word 
    def getDocNumContainT(self, word):
        res = 0
        if(self.vocab.has_key(word)):
            doc_tf_dict = self.vocab[word]
            for doc in doc_tf_dict:
                if(doc_tf_dict[doc] > 0):
                    res += 1
        return res 

    #return total documents in corpus
    def getDocNum(self):
        word = self.vocab.popitem()[0]
        return len(self.vocab[word])


'''read raw text file'''
f = open(pathToCorpus)
for line in f:
    tokens =  word_tokenize(line)
    doc_name = tokens[0]
    doc_content = tokens[1:]
    #put everything in doc_content in a dict
    #....
    #for each word in vocab
    #    tf = count(word, document)
f.close()


