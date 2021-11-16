from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class Index:

    '''index is a dict
       {word_1 : {doc_id_1 : tf, doc_id_2 : tf}
        word_2 : {doc_id_1 : tf}, ...} ,
         ....  : .... }
    '''
    #TODO: Stemming
    def __init__(self, input):
        self.input = input
        self.stemmer = PorterStemmer()
        self.doc_num = 0
        self.vocab = {}
        self.doc_ids = {}
        self.num_to_doc = {}
        self.doc_length = {}
        self.processIndex()

    def processIndex(self):
        '''read raw text file'''
        with open(self.input, 'r') as f:
            doc_num = 0
            #each line is a document
            for line in f:
                #this is from nltk library
                tokens =  word_tokenize(line)
                doc_id = tokens[0]
                self.doc_ids[doc_id] = doc_num
                self.num_to_doc[doc_num] = doc_id
                doc_content = tokens[1:]
                # doc length
                self.doc_length[doc_num] = len(tokens) - 1
                doc_num += 1
                for word in doc_content:
                    #stem the word 
                    word = self.stemmer(word) 
                    #then proceed to add to vocab
                    if(self.vocab.__contains__(word)):
                        if(self.vocab[word].__contains__(doc_id)):
                            self.vocab[word][doc_id] += 1
                        else:
                            self.vocab[word][doc_id] = 1
                    else:
                        self.vocab[word] = {doc_id : 1}
            #for collection stats
            self.doc_num = doc_num

    def getIndex(self):
        return self.vocab

    #write index into an output file in format: word doc_id tf
    def writeToFile(self, output):
        with open(output, 'w') as w:
            for word in self.vocab:
                doc_ids = self.vocab[word]
                for doc_id in doc_ids:
                    tf = doc_ids[doc_id]
                    w.write('{} {} {}\n'.format(word,doc_id, tf))

    #return word tf in a corpus
    def getTFinC(self, word):
        if(self.vocab.__contains__(word)):
            return sum(self.vocab[word].values())
        else:
            return 0

    #return word tf in a given document
    def getTFinD(self, word, doc_id):
        if(self.vocab.__contains__(word) and self.vocab[word].__contains__(doc_id)):
            return self.vocab[word][doc_id]
        else:
            return 0

    #return total documents in corpus containing word
    def getDocNumContainT(self, word):
        res = 0
        if(self.vocab.__contains__(word)):
            doc_tf_dict = self.vocab[word]
            for doc in doc_tf_dict:
                if(doc_tf_dict[doc] > 0):
                    res += 1
        return res

    #return total documents in corpus
    def getDocNum(self):
        return self.doc_num




