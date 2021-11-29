import matplotlib.pyplot as plt

import buildindex
import vsm
import lsi
import pyterrier as pt
if not pt.started():
    pt.init()

pathToCorpus = './nfcorpus/test.docs'
pathToQuery = './nfcorpus/test.all.queries'
pathToQrels = './nfcorpus/test.3-2-1.qrel'
print('Building index')
idx = buildindex.Index(pathToCorpus)

vsm_results = []
lsi_results = []


bs = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for b in bs:
    vsm_model = vsm.VSM(idx, 'tfnorm', None, b)
    pathToResults = './results_test_tfnorm_' + str(b) + '_vsm'

    with open(pathToResults, 'w') as results:
        with open(pathToQuery, 'r') as queries:
            for q in queries:
                qid = q.split()[0]
                res, scores = vsm_model.evaluateQuery(q, 1000)
                rank = 1
                for doc in res:
                    results.write("{} Q0 {} {} {} test_tfnorm\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                    rank += 1
    res = pt.io.read_results(pathToResults)
    qrels = pt.io.read_qrels(pathToQrels)
    print(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))
    vsm_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))

    lsi_model = lsi.LSI(vsm_model, 1500)
    print('LSI Model')
    pathToResults = './results_test_tfnorm_' + str(b) + '_lsi'

    with open(pathToResults, 'w') as results:
        with open(pathToQuery, 'r') as queries:
            for q in queries:
                qid = q.split()[0]
                res, scores = lsi_model.evaluateQuery(q, 1000)
                rank = 1
                for doc in res:
                    results.write("{} Q0 {} {} {} test_tfnorm_lsi\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                    rank += 1

    res = pt.io.read_results(pathToResults)
    qrels = pt.io.read_qrels(pathToQrels)
    lsi_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))

map_lsi = []
ndcg_lsi = []
for i in range(len(bs)):
    map_lsi.append(lsi_results[i]['map'])
    ndcg_lsi.append(lsi_results[i]['ndcg'])

map_vsm = []
ndcg_vsm = []
for i in range(len(bs)):
    map_vsm.append(vsm_results[i]['map'])
    ndcg_vsm.append(vsm_results[i]['ndcg'])

plt.plot(bs, map_lsi)
plt.xlabel('b term in TFNORM')
plt.ylabel('MAP')
plt.title('LSI Using TFNORM in TFIDF')
plt.show()

plt.plot(bs, ndcg_lsi)
plt.xlabel('b term in TFNORM')
plt.ylabel('NDCG')
plt.title('LSI Using TFNORM in TFIDF')
plt.show()

plt.plot(bs, map_vsm)
plt.xlabel('b term in TFNORM')
plt.ylabel('MAP')
plt.title('VSM Using TFNORM')
plt.show()

plt.plot(bs, ndcg_vsm)
plt.xlabel('b term in TFNORM')
plt.ylabel('NDCG')
plt.title('VSM Using TFNORM')
plt.show()