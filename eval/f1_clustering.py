import warnings
warnings.filterwarnings("ignore")
import math
import os.path
import random
import sys
import time

####
## Adapted from: https://github.com/cmdevries/ClusterEval/blob/master/cluster_eval.py
## All rights belong to original creator
###
def f1(doc2clust, doc2cat):

	# Use document IDs from ground truth doc2cat, as it may contain less documents than doc2clust.
	docids = list(doc2cat.keys()) #verbs

	# calculate true positivies, false negative and false positives for all unique pairs of documents where are x,y and y,x are NOT unique 
	start = time.time()
	tp = 0
	fn = 0
	fp = 0
	document_count = len(docids);
	for i in range(document_count):
		#if (i + 1) % 1000 == 0:
		#    print '%d of %d : %s seconds' % (i, document_count, time.time() - start)
		
		for j in  range(i, document_count): #si no es el mismo verbo
			if i != j:
				#print docids[i], docids[j]
				# for multi label submissions treat each label as an example for tp, fn, fp
				# TODO(cdevries): this does not matter for SED2013 as it is single label, but how should this be treated for multi label? Should it be based on set intersections between documents? i.e. tp for all categories that match (set intersection) and then work out fp and fn for the rest 
				
				
				for docid_i_cluster in doc2clust[docids[i]]: #clase automatica v1

					for docid_j_cluster in doc2clust[docids[j]]: #clase automatica v2
						
						for docid_i_category in doc2cat[docids[i]]: #clase real v1
							for docid_j_category in doc2cat[docids[j]]: #clase real v2
								
								

								if docid_i_cluster == docid_j_cluster and docid_i_category == docid_j_category:
									tp += 1

									#print 'tp'

								elif docid_i_cluster == docid_j_cluster and docid_i_category != docid_j_category:
									fp += 1

									#print 'fp'
									
								elif docid_i_cluster != docid_j_cluster and docid_i_category == docid_j_category:
									fn += 1

									#print 'fn'

								#print('tp = %d, fn = %d, fp = %d' % (tp, fn, fp))
	try:
		recall = float(tp)/(tp+fp)
	except ZeroDivisionError:
		recall = 0   
	
	try:
		precision = float(tp)/(tp+fn)
	except ZeroDivisionError:
		precision = 0
		
	#print recall, precision
	numerator = 2.0*precision*recall
	denominator = precision+recall             
	if denominator < 1e-15:
		score = 0
	else:
		score = numerator / denominator
	#print score
	return score

