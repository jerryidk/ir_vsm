from nltk.tokenize import word_tokenize
pathToCorpus = '/home/jerry/workspace/cs_6550/proj/nfcorpus/dev.docs'

'''Inverted Index class 
    class index
        init
        get : word -> (doc_id, tf)
'''

'''read raw text file'''
f = open(pathToCorpus)
for line in f:
    doc_name = word_tokenize(line)[0]
f.close()


