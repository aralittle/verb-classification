#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'lara'
import os
import pandas
import argparse
import pickle
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, dice
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
import kernel_kmeans




def DoKernel_kmeansNclass(NClass,orig_data, typeData, dist):
	'''
	Performs kernel kmeans for a given number of classes
	:param NClass: number of classes
	:param orig_data: data (pandas dataframe)
	:param typeData: bin or probabilities
	:param dist: distance to be used
	:return: dictionary {number of classes_sh: [verb labels]}
	'''

	res = {} #initialize output

	labels = []
	if typeData == 'bin':
		X = pairwise_distances(orig_data, metric=dice)
		km = kernel_kmeans.KernelKMeans(n_clusters=NClass, max_iter=1000, random_state=0, kernel='precomputed', verbose=1)
		labels = km.fit_predict(X)
		print('number of classes ',len(set(labels)))
	
	else:
		try:
			km = kernel_kmeans.KernelKMeans(n_clusters=NClass, max_iter=1000, random_state=0, kernel=dist, verbose=1)
			labels = km.fit_predict(orig_data)
			print('number of classes ',len(set(labels)))
		except Exception as e:
			print(e)

	if not labels == []:
		num_clases = len(set(labels))
		if 1 < num_clases < orig_data.shape[0]: #between 2 and number of verbs-1 classes
			try:
				sh = silhouette_score(orig_data, labels, dist)
			except:
				sh = 'NoSh'
			if not num_clases in res:
				if not type(sh) == str:
					k = str(num_clases) + '_' + str(round(sh, 3))
				else:
					k = str(num_clases) + '_NoSh'
				res[k] = labels
			else:
				print('number of classes already obtained')

	return res


def DoAgglo(orig_data, linked, metric):
	'''
	Performs agglomerative clustering and formats the result.
	:param orig_data: matrix of m*n (senses*features) from pandas.read_csv
	:param linked: distance matrix obtained with linkage method (scipy.cluster.hierarchy)
	:param metric: cosine or dice
	:return: Dic object -> {numClasesofClustering_sh:[labels], ...} --> {srt(Int)_str(Float):[int,int,int], ...}
	Labels is the list of the clusterId that corresponds to each sense
	'''

	result = {} #initialize output

	try:
		cut_vals = cophenet(linked) # distance cut values that create the groups
		cut_set = set(cut_vals)
		ord_list = sorted(list(cut_set))
		for o in ord_list:  ## for each possible cutpoint, obtain the clustering

			labels = fcluster(linked, o, criterion='distance')
			num_clases = len(set(labels))
			if 1 < num_clases < orig_data.shape[0]/2: #between 2 and num of verbs-1 number of classes/2
				sh = silhouette_score(orig_data, labels, metric=metric)
				if not num_clases in result:
					k = str(num_clases) + '_' + str(round(sh, 3))
					result[k] = labels
				else:
					print('number of classes already obtained')
	except ValueError:
		# Not working for that cut value
		pass

	return result


def DoAggloNumClass(orig_data, linked, metric, Nclas):
	'''
	Performs agglomerative clustering for a specific number of classes and formats the result.
	:param orig_data: matrix of m*n (senses*features) from pandas.read_csv
	:param linked: distance matrix obtained with linkage method (scipy.cluster.hierarchy)
	:param metric: cosine or dice
	:param Nclas: a given number of classes for the clustering
	:return: Dic object -> {numClasesofClustering_sh:[labels], ...} --> {srt(Int)_str(Float):[int,int,int], ...
	Only one element in this case
	Labels is the list of the clusterId that corresponds to each sense
	'''


	res = {} #initialize output

	labels = fcluster(linked, Nclas, criterion='maxclust')
	num_clases = len(set(labels))
	if str(num_clases) == str(Nclas):
		if 1 < num_clases < orig_data.shape[0]: #between 2 and and num of verbs-1 number of classes

			sh = silhouette_score(orig_data, labels, metric=metric)
			if not num_clases in res:
				k = str(num_clases) + '_' + str(round(sh, 3))
				res[k] = labels
			else:
				print('number of classes already obtained')

	return res


def resultToDic(dataframe, clusterid):
	'''
	Maps the result (ClusterIds for each sense) with the lemmas of the senses
	:param dataframe: dataframe pandas object obtained from the training file
	:param clusterid: clustering, format is output of DoAgglo and DoAggloNumClass
	:return: dic object {cluster:[verbos], ...}
	'''

	dictCluster = {}
	r = dataframe.iterrows()
	objects = [e[0] for e in r] #e[0] is the verb sense
	for index in range(len(clusterid)):
		c = clusterid[index]  # cluster id for that sense
		o = objects[index]  # corresponding sense
		if c in dictCluster:
			dictCluster[c].append(o)
		else:
			dictCluster[c] = [o]
	return dictCluster



def mainAgglo(file, out, Nclass, methods):
	'''
	Executes the cluster algorithm over the data. Generates all possible clusterings per data file
	Stores the output in a pickle object
	:param file: input, file that contains training data
	:param out: directory that will contain output file that holds the pickled data
	:return: nothing
	Format of output: {'agglo_NumClusters_method_metric_sh':({ncluster:[senses]}, [original labels])}
	'''


	outname = file.split('/')[-1]
	with open(os.path.join(out, outname), 'wb') as results:
		print('performing agglomerative clustering on ', outname)
		orig_data = pandas.read_csv(file, index_col=0, header=0)


		clustering_dic = {}
		#only coherent metrics and data formats
		if 'bin' in file:
			met = 'dice'
		else:
			met = 'cosine'

		for md in methods:
			linked = linkage(orig_data, method=md, metric=met)
			print('clustering...')

			if Nclass is None: #explore all partitions
				n_labels = DoAgglo(orig_data, linked, metric=met) #{numClasesofClustering_sh:[labels], ...}
			else:
				n_labels = DoAggloNumClass(orig_data, linked, met, Nclass)

			for n in n_labels:
				dic = resultToDic(orig_data, n_labels[n])
				numClusters = int(n.split('_')[0])
				sh = float(n.split('_')[1])
				originalLabels = n_labels[n]
				metric = met
				method = md
				key = 'agglo_' + '{0}_'.format(numClusters) + '{0}_'.format(method) + '{0}_'.format(
					metric) + '{0}'.format(str(sh))
				clustering_dic[key] = (dic, originalLabels)
		pickle.dump(clustering_dic, results)




def mainKernel(file, out, metricsKernel, numClusters):
	'''
	Executes the cluster algorithm over the data. Generates all possible clusterings per data file
	Stores the output in a pickle object
	:param file: input, file that contains training data
	:param out: output directory for file that contains the pickled data
	:param metricsKernel: list of metrics that can be tried
	:param numClusters: int, number of clusters
	:return: nothing
	Format of output: {'agglo_NumClusters_method_metric_sh':({ncluster:[senses]}, [original labels])}
	'''
	
	#for fi in os.listdir(dir1): # training files
	outname = file.split('/')[-1]
	with open(os.path.join(out, outname), 'wb') as results:
		print('performing kernel kmeans on ', outname)
		orig_data = pandas.read_csv(file, index_col=0, header=0)

		clustering_dic = {}
		if 'bin' in file:

			n_labels = DoKernel_kmeansNclass(numClusters, orig_data.values, 'bin', dice)
			for n in n_labels: #for each possible clustering with the parameters (metric and method)
				dic = resultToDic(orig_data, n_labels[n])  # {ncluster:[senses]}
				numClusters = int(n.split('_')[0])
				sh = float(n.split('_')[1])
				originalLabels = n_labels[n]
				metric = 'dice'
				method = 'noMethod'
				key = 'kernel_{0}_{1}_{2}_{3}'.format(numClusters, method, metric, str(sh))
				clustering_dic[key] = (dic,originalLabels)

		else:

			for met in metricsKernel: #try the different metrics
				print('using metric ', met)
				n_labels = DoKernel_kmeansNclass(numClusters, orig_data.values, 'nobin', met)

				for n in n_labels: #for each possible clustering with the parameters (metric and method)
					dic = resultToDic(orig_data, n_labels[n])  # {ncluster:[senses]}
					numClusters = int(n.split('_')[0])
					sh1 = n.split('_')[1]
					if not type(sh1) == str:
						sh = float(sh1)
					else:
						sh = sh1

					originalLabels = n_labels[n]
					metric = met
					method = 'noMethod'
					key = 'kernel_{0}_{1}_{2}_{3}'.format(numClusters, method, metric, str(sh))
					clustering_dic[key] = (dic,originalLabels)

		pickle.dump(clustering_dic, results)

		

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--selection', type=list, default=None, nargs='*', help='List of files to perform clustering')
	parser.add_argument('-i','--input', help='Input folder where csv files are', default='../GeneratedData/training/')
	parser.add_argument('-o','--output', help='Output folder to materialize clusterings', default='../clusterings/2ndBatch/')
	parser.add_argument('-a', '--algorithm', choices=['hierarchical','kernelKmeans'], default = 'hierarchical',help = 'Clustering algorithm')
	parser.add_argument('-n','--numberOfClasses', default=None, help='Number of classes for the algorithm')

	args = parser.parse_args()

	metrics = ['cosine', 'dice']
	methods = ['single', 'average', 'complete']  # to use centroic distance needs to be euclidean
	metricsKernel = ['polynomial', 'laplacian', 'linear', 'additive_chi2', 'sigmoid', 'chi2', 'rbf', 'cosine', 'poly']

	for file in os.listdir(args.input): # training files
		if file in args.selection[0] or args.selection == None:
			print('clustering ', file)
			if args.algorithm == 'hierarchical':
				if args.numberOfClasses is not None:
					print("Clustering with {0} classes".format(str(args.numberOfClasses)))
				else:
					print("No number of classes was specified. Performing clustering for num of classes "
						  "in range 2-n-1 number of elements")
				mainAgglo(os.path.join(args.input, file), args.output, args.numberOfClasses, methods)
			if args.algorithm == 'kernelKmeans':
				if args.numberOfClasses is not None:
					mainKernel(os.path.join(args.input, file), args.output, metricsKernel, int(args.numberOfClasses))
				else:
					print('Number of classes needs to be specified (-n Number)')


if __name__ == "__main__":
	main()

