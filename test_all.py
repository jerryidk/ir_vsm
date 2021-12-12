import matplotlib.pyplot as plt

import buildindex
import vsm
import lsi
import pyterrier as pt
if not pt.started():
    pt.init()

pathToCorpus = './nfcorpus/test.docs'
pathToQuery = './nfcorpus/test.titles.queries'
pathToQrels = './nfcorpus/test.3-2-1.qrel'

print('Building index')
idx = buildindex.Index(pathToCorpus)
vsm_model = vsm.VSM(idx, 'tf', 'idf', 'standard', numpy_file='./test_tf_idf_stand.npz')

vsm_results = []
pathToResults = './results_test_tf_vsm'
with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res, scores = vsm_model.evaluateQuery(q, 1000)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} {} test_tf_vsm\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                rank += 1

res = pt.io.read_results(pathToResults)
qrels = pt.io.read_qrels(pathToQrels)
vsm_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))
print('tf idf standard')
print(vsm_results)

vsm_model = vsm.VSM(idx, 'tf', 'idf', 'standard', numpy_file='./test_tf_idf_none.npz')

vsm_results = []
pathToResults = './results_test_tf_vsm'
with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res, scores = vsm_model.evaluateQuery(q, 1000)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} {} test_tf_vsm\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                rank += 1

res = pt.io.read_results(pathToResults)
qrels = pt.io.read_qrels(pathToQrels)
vsm_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))
print('tf idf none')
print(vsm_results)

vsm_model = vsm.VSM(idx, 'tf', 'idf', 'standard', numpy_file='./test_tf_squareidf_stand.npz')

vsm_results = []
pathToResults = './results_test_tf_vsm'
with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res, scores = vsm_model.evaluateQuery(q, 1000)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} {} test_tf_vsm\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                rank += 1

res = pt.io.read_results(pathToResults)
qrels = pt.io.read_qrels(pathToQrels)
vsm_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))
print('tf suareidf standard')
print(vsm_results)

vsm_model = vsm.VSM(idx, 'tf', 'idf', 'standard', numpy_file='./test_tf_squareidf_none.npz')

vsm_results = []
pathToResults = './results_test_tf_vsm'
with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res, scores = vsm_model.evaluateQuery(q, 1000)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} {} test_tf_vsm\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                rank += 1

res = pt.io.read_results(pathToResults)
qrels = pt.io.read_qrels(pathToQrels)
vsm_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))
print('tf squareidf none')
print(vsm_results)

vsm_model = vsm.VSM(idx, 'tf', 'idf', 'standard', numpy_file='./test_tflog_idf_stand.npz')

vsm_results = []
pathToResults = './results_test_tf_vsm'
with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res, scores = vsm_model.evaluateQuery(q, 1000)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} {} test_tf_vsm\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                rank += 1

res = pt.io.read_results(pathToResults)
qrels = pt.io.read_qrels(pathToQrels)
vsm_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))
print('tflog idf standard')
print(vsm_results)

vsm_model = vsm.VSM(idx, 'tf', 'idf', 'standard', numpy_file='./test_tflog_idf_none.npz')

vsm_results = []
pathToResults = './results_test_tflog_vsm'
with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res, scores = vsm_model.evaluateQuery(q, 1000)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} {} test_tf_vsm\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                rank += 1

res = pt.io.read_results(pathToResults)
qrels = pt.io.read_qrels(pathToQrels)
vsm_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))
print('tf idf none')
print(vsm_results)

vsm_model = vsm.VSM(idx, 'tf', 'idf', 'standard', numpy_file='./test_tflog_squareidf_stand.npz')

vsm_results = []
pathToResults = './results_test_tflog_vsm'
with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res, scores = vsm_model.evaluateQuery(q, 1000)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} {} test_tf_vsm\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                rank += 1

res = pt.io.read_results(pathToResults)
qrels = pt.io.read_qrels(pathToQrels)
vsm_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))
print('tf suareidf standard')
print(vsm_results)

vsm_model = vsm.VSM(idx, 'tf', 'idf', 'standard', numpy_file='./test_tflog_squareidf_none.npz')

vsm_results = []
pathToResults = './results_test_tflog_vsm'
with open(pathToResults, 'w') as results:
    with open(pathToQuery, 'r') as queries:
        for q in queries:
            qid = q.split()[0]
            res, scores = vsm_model.evaluateQuery(q, 1000)
            rank = 1
            for doc in res:
                results.write("{} Q0 {} {} {} test_tf_vsm\n".format(qid, doc, rank, scores[vsm_model.index.doc_ids[doc]]))
                rank += 1

res = pt.io.read_results(pathToResults)
qrels = pt.io.read_qrels(pathToQrels)
vsm_results.append(pt.Utils.evaluate(res, qrels, metrics = ['map', 'ndcg']))
print('tf squareidf none')
print(vsm_results)