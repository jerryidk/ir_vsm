import buildindex
import vsm
import lsi

pathToCorpus = '../nfcorpus/dev.docs'
pathToQuery = '../nfcorpus/dev.all.queries'
pathToResults = './results'
idx = buildindex.Index(pathToCorpus)
model = vsm.VSM(idx, 'tflog', './dev_tflog_tfidf.npy')
model = lsi.LSI(model)

with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res = model.evaluateQuery(q, 10)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} -Infinity galago\n".format(qid, doc, rank))
                rank += 1


