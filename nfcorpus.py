import buildindex
import vsm

pathToCorpus = './nfcorpus/dev.docs'

idx = buildindex.Index(pathToCorpus)
model = vsm.VSM(idx, 'tflog')
print(model.evaluateQuery('why deep fried foods may cause cancer', 10))