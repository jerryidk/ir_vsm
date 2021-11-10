-------------------------------------------------------------
buildindex.py : Build inverted index  
Description: 
Take corpus (eg. dev.docs), then output a index file in format
where each line follows

word doc_id tf  

example:  

informaiton MED-3140 10  

meaning word information has tf 10 in document 3140.

To run buildindex.py 
`python3 buildindex.py <inputfile> <indexfile>`
---------------------------------------------------------------
vsm.py : compute similarity between query and doc.
Description:
Take a index file and query file (e.g. dev.title.queries) and parameter file
and output a retrieval result file.

parameter file should be in following format  

parameter_name        options   

weight_func        tf, tf_log, tf_norm  

documents_returns      `<int>`  

.....                   .....  


retrival result file should be in following format
query-id document-id score

-----------------------------------------------------------------
eval.py : evaluate performance of the model
Description: 
Takes a retrieval result file and judgement file (e.g. qrel), and output a evaluation file

evaluation file format

query-id ap  

....     ...  

....     ...  

map ...




