import buildindex
import vsm
import lsi

pathToCorpus = './nfcorpus/test.docs'
pathToQuery = './nfcorpus/test.all.queries'
pathToResults = './results_test_tf_vsm'
pathToResults = None
print('Building index')
idx = buildindex.Index(pathToCorpus)
model = vsm.VSM(idx, 'tf')

with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res = model.evaluateQuery(q, 10)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} -Infinity galago\n".format(qid, doc, rank))
                rank += 1

model = lsi.LSI(model)
pathToResults = './results_test_tf_lsi'

with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res = model.evaluateQuery(q, 10)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} -Infinity galago\n".format(qid, doc, rank))
                rank += 1