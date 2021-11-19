import buildindex
import vsm

pathToCorpus = '../nfcorpus/dev.docs'
pathToQuery = '../nfcorpus/dev.all.queries'
pathToResults = './results'
idx = buildindex.Index(pathToCorpus)
model = vsm.VSM(idx, 'tflog')

with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res = model.evaluateQuery(q, 3)
            rank = 1
            for doc in res:
                results.write("{}, Q0, {}, {}, galago\n".format(qid, doc, rank))
                rank += 1


