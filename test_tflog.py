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
vsm_model = vsm.VSM(idx, 'tflog')

vsm_results = []
pathToResults = './results_test_tflog_vsm'
with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res, scores = vsm_model.evaluateQuery(q, 1000)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} {} test_tflog\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                rank += 1

res = pt.io.read_results(pathToResults)
qrels = pt.io.read_qrels(pathToQrels)
vsm_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))
print(vsm_results)

print('Building LSI')
lsi_results = []
topics = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
for topic in topics:
    print('topic: ' + str(topic))
    lsi_model = lsi.LSI(vsm_model, topic)
    pathToResults = './results_test_tflog_lsi_topics_' + str(topic)

    with open(pathToResults, 'w') as results:
        with open(pathToQuery, 'r') as queries:
            for q in queries:
                qid = q.split()[0]
                res, score = lsi_model.evaluateQuery(q, 1000)
                rank = 1
                for doc in res:
                    results.write("{} Q0 {} {} {} test_tflog_lsi\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                    rank += 1

    res = pt.io.read_results(pathToResults)
    qrels = pt.io.read_qrels(pathToQrels)
    lsi_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))

print(lsi_results)

map_lsi = []
ndcg_lsi = []
for i in range(len(topics)):
    map_lsi.append(lsi_results[i]['map'])
    ndcg_lsi.append(lsi_results[i]['ndcg'])

plt.plot(topics, map_lsi)
plt.xlabel('Number of Topics')
plt.ylabel('MAP')
plt.title('LSI Using TFLOG in TFIDF')
plt.show()

plt.plot(topics, ndcg_lsi)
plt.xlabel('Number of Topics')
plt.ylabel('NDCG')
plt.title('LSI Using TFLOG in TFIDF')
plt.show()