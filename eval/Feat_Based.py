# -*- coding: utf-8 -*-
'''
Created on 29/06/2016

@author: lara

'''

from collections import  Counter
from scipy.spatial.distance import dice, cosine
import csv, numpy


def csvToDic(file1):
	'''
	Converts a csv containing a matrix into a dict that contains info about the weight of each feature in each verb
	:param file1: csv file containing data
	:return: dict object {verb:{feat:wight, feat:weight}, verb:{...}}
	'''
	dic = {}
	with open(file1, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		cats = []
		for row in reader:
			if row[0] == 'head':
				cats = row[1:] #feat categories
			else:#it is a sense
				feats = {}
				verb = row[0]
				values = row[1:]
				for e in range(len(values)):
					if values[e] not in ['0','0.0',0,0.0]:
						feats[cats[e]] = float(values[e])

				dic[verb] = feats
	return dic


def getAllFeats(file1):
	'''
	Obtain a list of the features that are being used
	:param file1: csv file containing data
	:return: list containing names of features (strings) ['feat1','feat2', etc]
	'''
	with open(file1, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			if row[0] == 'head':
				cats = row[1:]
				return cats

def getVerbFeat(verb,feat,dicValues):
	'''
	Given a verb and a feature, recovers its value from featuredic
	:param verb: a sense (str)
	:param feat: a feature (str)
	:param dicValues: dict object that contains info about the weight of each feature for each sense (from csvToDic)
	:return: weight of feature
	'''
	Rfeats = dicValues[verb]
	if feat in Rfeats:
		return Rfeats[feat]
	else:
		return 0


def ClustersSize(clustering):
	'''
	Creates a dict object from the clustering object with info about the number of verbs in each cluster
	:param clustering: clustering system
	:return: dict {ClusterID:numVerbs, ...}
	'''
	d= Counter()
	for c, verbs in clustering.iteritems():
		d[c]=len(verbs)
	return d

def Equal_biggerThan(dicValues, clustering):
	'''
	Creates a dict object that contains information about the number of members that each cluster has
	:param dicValues: dict object that contains info about the weight of each feature for each sense (from csvToDic)
	:param clustering: clustering system
	:return: dict object {MoreOrEqualNumberofSenses: [clusterID that has this number or +, ClusterID,..],..}
	'''
	dic = {}
	for elem in range(1,len(dicValues)+1): #for 1-N number of senses
		for c in clustering:
			if len(clustering[c]) > elem or  len(clustering[c]) == elem: #if clustering has less or equal number of senses
				if elem in dic:
					dic[elem].append(c)
				else:
					dic[elem]= [c]
	return dic
					
	
	

			
def getNumberBiggerClus(cluster, clustering):
	'''
	Returns the mumber of clusters in a clustering system that have more population than a given clusterID
	:param cluster: clusterID
	:param clustering: clustering system
	:return: Integer, number of clusters larger than given cluster
	'''
	'''
	devuelve el num de clusters más grandes que cluster
	'''
	bigger = 0
	clustersizes = ClustersSize(clustering)
	lenc = clustersizes[cluster]
	for c in clustersizes:
		if clustersizes[c] > lenc:
			bigger += 1
	return bigger
	


def WeightCF(clustering,dicValues):
	'''
	Calculates the weight of each feature in each cluster. all are added even if they have 0 weight
	:param clustering: cluster system
	:param dicValues: dict object that contains info about the weight of each feature for each sense (from csvToDic)
	:return: dict object {cluster{feat:weight, feat:weight}, cluster:{...}}
	'''

	dic = {}
	for c in clustering.iterkeys(): #cluster
		for v in clustering[c]: #sense
			dicfeats = dicValues[v]
			for feat, value in dicfeats.iteritems():
				if c in dic:
					if feat in dic[c]:
						dic[c][feat]+=float(value)
					else:
						dic[c][feat]=float(value)	
				else:
					dic[c] = {feat:float(value)}
	return dic


def WeightFC(clustering, dicValues,allFeats):
	'''
	Calculates the weight of each feature in each cluster
	:param file1: csv file
	:param clustering: clustering system
	:return: dict object {'feat1': {'c3': 0.4, 'c2': 0.3, 'c1': 0.2}}
	'''

	featDic = {}


	for f in allFeats:
		for c in clustering.iterkeys():
			vbs = clustering[c]
			for v in vbs:
				if f in dicValues[v]:	

					weight = dicValues[v][f]
					if f in featDic:
						if c in featDic[f]:
							featDic[f][c] += float(weight)
						else:
							featDic[f][c] = float(weight)
					else:
						featDic[f] = {c:float(weight)}
	return featDic	


	
def GetlocalRecall(feature, cluster, clustering, dicValues):
	'''

	Calculate local recall for a feature and a cluster
	:param feature: feature (str)
	:param cluster: cluster Id (int)
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:return: weight of feat in cluster/ weight of feat in clustering or None
	'''

	WeightFinC = 0 # weight of feature F in (all verbs of) a cluster
	for v in clustering[cluster]:
		valFeatinV = getVerbFeat(v,feature,dicValues) #value of feature for verb
		WeightFinC += float(valFeatinV)
	if WeightFinC == 0:
		print 'Recall: cluster {0} has no feature {1}'.format(str(cluster),feature)
		return None
	
	TotalF = 0 # weight of feature F in all groups of clustering
	for c in clustering:
		for v in clustering[c]:
			valFeatinV = getVerbFeat(v,feature,dicValues)
			TotalF +=float(valFeatinV)
	#print feature, 'c',cluster, TotalF
	if TotalF == 0:

		print 'Recall: no se encontro {0} en clustering'.format(feature)
		return None
	#print 'recall',feature, cluster, WeightFinC, float(TotalF),  WeightFinC/ float(TotalF)
	return WeightFinC/ float(TotalF)


def GetlocalPrecision(feature, cluster, clustering, dicValues):
	'''
	Precision is good if each cluster has a majoritary feature
	(for labeling is not necessary)
	Calculate local precision for a feature and a cluster
	(weight of f in c/weight all f in c)
	:param feature: feature, string
	:param cluster: clusterID, int
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:return:
	'''

	WeightFinC = 0 #weight feature f in cluster
	for v in clustering[cluster]:
		valFeatinV = getVerbFeat(v,feature,dicValues)
		WeightFinC += float(valFeatinV)
	if WeightFinC == 0:

		print 'Precision: no se encontro {0} en cluster {1}'.format(feature, str(cluster))
		return None
		
	AllFinC = 0 #all feats of a cluster
	for v in clustering[cluster]:
		for f in dicValues[v].iterkeys():
			weight = getVerbFeat(v,f,dicValues)
			AllFinC += float(weight)

	if AllFinC == 0:
		print 'Precision: no features in cluster {0}'.format(str(cluster))
		return None
	return WeightFinC/ float(AllFinC)	
				

def Fmeasurelocal(local_prec, local_rec):
	'''
	Calculate F-measure local
	:param local_prec: local precission
	:param local_rec: local recall
	:return: F-measure, float
	'''
	if local_prec ==0 and local_rec ==0:
		return 0
	return 2*float(local_rec*local_prec)/float(local_rec+local_prec)



def MacroRecall(clustering, dicValues, maxFeats):
	'''
	Calculate macro recall (average recall del clustering). Deppends on the amount of clusters. Should be high, together with
	macroprecission. Includes and punishes clusters that do not have maximal features
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:param file1: csv file
	:param flag: binary:'max' or 'nomax'. if 'max' setmaximalFeatsNew to determine maximal features is used.
	Else, clusterFeats is used, it identifies al features that are not 0 in the cluster
	:return: float or None
	'''

	if maxFeats == {}:
		return None

	total = 0
	for cluster, featlist in maxFeats.iteritems():
		#print(cluster, featlist)

		tempF = 0 #recall of all maxfeats of a cluster
		n=0 #number of max feats with value
		for feat in featlist:
			Localrecall = GetlocalRecall(feat, cluster, clustering, dicValues)
			if Localrecall:
				tempF += Localrecall
				n+=1
		if len(featlist) != 0 and n!= 0:
			dC = tempF/float(n) #average recall of max feats for a cluster
		else:
			#no max feats for this cluster
			dC = 0
		total += dC
	return total/float(len(maxFeats))
			


def MacroPrecision(clustering, dicValues,maxFeats):
	'''
	Calculates average precission of clustering. Should be high
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:param file1: csv file
	:param flag:  binary:'max' or 'nomax'. if 'max' setmaximalFeatsNew to determine maximal features is used.
	Else, clusterFeats is used, it identifies al features that are not 0 in the cluster
	:return: float or None
	'''

	if maxFeats == {}:
		return None

	total = 0
	for cluster, featlist in maxFeats.iteritems():
		tempF = 0 #f-average of main features
		n = 0 #features in cluster
		for feat in featlist:
			Localprecision = GetlocalPrecision(feat, cluster, clustering, dicValues)
			if Localprecision:
				tempF += Localprecision
				n+=1

		if len(featlist) != 0 and n != 0:
			dC = tempF/float(n)
		else:
			dC = 0
		
		total += dC

	return total/float(len(maxFeats))#average precission for features in clustering
	#includes and punishes clusters that do not have relevan features




def FmeasureMacro(macro_prec, macro_rec):
	'''
	Calculates F measure
	:param macro_prec: macro precission
	:param macro_rec: macro recall
	:return: F measure, float
	'''
	if macro_prec ==0 and macro_rec ==0:
		return 0
	return 2*float(macro_rec*macro_prec)/float(macro_rec+macro_prec)

def MicroRecall(clustering, dicValues,maxFeats):
	'''
	average recall of clustering divided by the amount of features independently of the structure of the cluster
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:param file1: csv file
	:param flag: binary:'max' or 'nomax'. if 'max' setmaximalFeatsNew to determine maximal features is used.
	Else, clusterFeats is used, it identifies al features that are not 0 in the cluster
	:return: float
	'''


	total = 0
	for cluster, featlist in maxFeats.iteritems():
		for feat in featlist:
			Localrecall = GetlocalRecall(feat, cluster, clustering, dicValues)
			total += Localrecall
	return total/float(len(dicValues)) #divided by the amount of features
			


def MicroPrecision(clustering, dicValues,maxFeats):
	'''
	Average precission of clustering devided by the number of features
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:param file1: csv file
	:param flag: binary:'max' or 'nomax'. if 'max' setmaximalFeatsNew to determine maximal features is used.
	Else, clusterFeats is used, it identifies al features that are not 0 in the cluster
	:return: float
	'''

	total = 0
	for cluster, featlist in maxFeats.iteritems():
		for feat in featlist:
			Localprecision = GetlocalPrecision(feat, cluster, clustering, dicValues)
			total += Localprecision
	return total/float(len(dicValues))


def FmeasureMicro(micro_prec, micro_rec):
	'''
	Fmeasure for microprecission and microrecall
	:param micro_prec: micro precission
	:param micro_rec:  micro recall
	:return: float
	'''
	if micro_prec ==0 and micro_rec ==0:
		return 0
	return 2*float(micro_rec*micro_prec)/float(micro_rec+micro_prec)


				
def setmaximalFeats(clustering,dicValues, allFeats):
	'''
	Set maximal features for a cluster.
	A feature is said to be maximal for a cluster iff its weight is higher for that cluster than any other
	{cluster:{feat:Fmeasure,feat:Fmeasure}, cluster:...}
	:param file1: csv file
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:return: two dicts for weight and recall {cluster:[maxfeat1, maxfeat2]}
	'''

	# collect max feats:  {cluster: [feats]}
	maxFeats = {} #max weight
	maxFeatsRec = {} #max recall

	WeightFeatInCl = WeightFC(clustering, dicValues, allFeats) #{feat:{cl:val,cl:val},feat:{}}

	for f in allFeats:

		if f in WeightFeatInCl: #{cluster:weight, cluster:weight}

			#value of each feat for each cluster, delete values ==0
			search = { k : v for k,v in WeightFeatInCl[f].iteritems() if v != 0}

			#find cluster in which this feature has more weight
			maximCl = [key for key,val in search.iteritems() if val == max(search.values())] #returns a list

			for clu in maximCl:
				if clu in maxFeats:
					maxFeats[clu].append(f)
				else:
					maxFeats[clu] =[f]

			#find cluster for which this f has max recall
			temp = {}
			for c in search: #clusters of a feat where this feat is not 0

				R = GetlocalRecall(f, c, clustering, dicValues)
				temp[c] = R
				#print f, c, R


			maximCl2 = [key for key,val in temp.iteritems() if val == max(temp.values())]
			#print'max',f, maximCl2
			for clu in maximCl2:
				if clu in maxFeatsRec:

					maxFeatsRec[clu].append(f)
				else:
					maxFeatsRec[clu] =[f]
	

			#there are clusters that are left outside because their features have no recall for them
			'''			
			for c in clustering:
				if c not in maxFeats:
					total = 0
					for r in prueba[c]:
						total += prueba[c][r]
					print 'cluster sin peso' ,c, total
			'''

	return maxFeats, maxFeatsRec


			
def averageFM_fs_C(clustering, dicValues, allFeats):
	'''
	Calculates average Fmeasure for each feature for clusters in which it is not 0
	:param file1: csv
	:param clustering: clustering system
	:return: dict object
	'''

	AverageF_featDic = {}
	#allFeats = getAllFeats(file1)
	#dicValues = csvToDic(file1)
	CF = WeightCF(clustering,dicValues) #{c{f:0,f2.3}...}

	for f in allFeats: #in training

		weights = 0 #weight of f for all clusters
		ClustersConF = 0
		for c in clustering.iterkeys():
			if f in CF[c]:
				localRecall = GetlocalRecall(f, c, clustering, dicValues)
				localPrecision =  GetlocalPrecision(f, c, clustering, dicValues)
				if localPrecision and localRecall:
					fcFmeasure = Fmeasurelocal(localPrecision,localRecall)
					#print f,c,localRecall,localPrecision,fcFmeasure
					weights += fcFmeasure
					ClustersConF+=1
				#print 'f',f, fcFmeasure
		if ClustersConF == 0:
			res=0
		else:
			res = weights / float(ClustersConF) ## average f-measure of feature in clusters in which it is not 0
		AverageF_featDic[f] = res
	
	return AverageF_featDic
					
	
def averageFM_fs_global(clustering,dicValues,allFeats):
	'''
	Calculates f-measure average for clusters that contain maximal features
	:param file1: csv file
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:return: float, average f measure for all maximal features
	'''

	AverageF_featDic= averageFM_fs_C(clustering,dicValues, allFeats) #{f:averageFmeas}
	# average Fmeasure for each feature for clusters in which it is not 0 divided by the number of max features
	r = sum(AverageF_featDic.values())/float(len(AverageF_featDic))
	return r


def clusterFeats(clustering,dicValues):
	'''
	For each cluster, dentifies features that are not 0
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:return: dict {cluster:[feats]}
	'''

	out = {}
	CF = WeightCF(clustering,dicValues)
	#print 'CF', CF
	for c in CF:
		out[c]=[]
		for f in CF[c]:
			if CF[c][f] !=0:
				out[c].append(f)
	#print 'all cluster feats', out
	return out

def setmaximalFeatsNew(clustering,dicValues,allFeats):
	'''
	Get maximal features. A feature is said to be maximal for a cluster iff its Fmeasure for that cluster is higher
	than its average Fmeasure for all the clustering and the average Fmeasure of all the features
	:param clustering: clusterin system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:param allFeats: all the features of the cluster
	:return: dict {cluster:[feat, feat], cluster:...}
	'''

	maxFeats = {} # {cluster:[feat, feat], cluster:...}

	av = averageFM_fs_C(clustering,dicValues,allFeats) #'dic average fmeasure for each f'{f:fm} in all clusters that have it
	alG = averageFM_fs_global(clustering,dicValues,allFeats)#int 'average F measure of all features together'
	WeightFeatInCl = WeightCF(clustering,dicValues) #{feat:{cl:val,cl:val},feat:{}}
	#print 'feat per cl',WeightFeatInCl
	for c in clustering:
		#print 'max feats of', c
		for f in WeightFeatInCl[c]:
			#get f-measure of f
			localRecall = GetlocalRecall(f, c, clustering, dicValues)
			localPrecision =  GetlocalPrecision(f, c, clustering, dicValues)
			if localRecall and localPrecision:
				fcFmeasure = Fmeasurelocal(localPrecision,localRecall)
				#print 'ver',c, f, fcFmeasure, 'greater than',av[f], alG

				#print f,c,fcFmeasure
				if fcFmeasure >= av[f] and fcFmeasure >= alG: # Fmeasure bigger than average f-m for this feat all clusters
																# and than average of all features
					if c in maxFeats:
						maxFeats[c].append(f)
					else:
						maxFeats[c]=[f]
	for c in clustering:
		if c not in maxFeats:
			maxFeats[c]= []
	#print maxFeats
	return maxFeats





def contrast(clustering, dicValues, feature, cluster,allFeats):
	'''
	Selects active features, features relevant for cluster: determine if their representations.
	The contrast value is how good is the f measure of this feature comared to intracluster inertia
	(higher, better)
	#are better than average in this cluster
	:param file1: csv file
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:param feature: feature
	:param cluster: clusterID
	:return: float or none, f-measure of f for c/average f-measure of f in C
	'''


	fmDic = averageFM_fs_C(clustering,dicValues,allFeats) #'dic average fmeasure for each f'{f:fm} in all clusters that have it
	fvalinC = fmDic[feature] #f-measure de f en c

	localRecall = GetlocalRecall(feature, cluster, clustering, dicValues)
	localPrecision =  GetlocalPrecision(feature, cluster, clustering, dicValues)
	fcFmeasure = Fmeasurelocal(localPrecision,localRecall)
	if fvalinC != 0:
		#print 'valor de ', feature, fcFmeasure, 'valor medio en clustering ', fvalinC
		return fcFmeasure/float(fvalinC) #f-measure of f for c/average f-measure of f in C
	return None
	

def cumulativeMicroPrecision(clustering, dicValues,maxFeats):
	'''
	calculate microprecision cumulatively
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:param maxFeats: more important features
	:return: float
	'''


	eb = Equal_biggerThan(dicValues, clustering) #{NumberMembers:[ClusterID, clusterID]}
	#print 'eb', eb
	temp2 = 0
	for Nelems,listaClusters in eb.iteritems():
		sumaC = 0	#accumulates precission of clusters
		for cluster in listaClusters:
			for f in maxFeats[cluster]:
				prec = GetlocalPrecision(f, cluster, clustering, dicValues)	
				sumaC += prec
				#print prec
		#possible problem: actually clusters that have few members and many features are favored
		temp = float(sumaC)/(len(listaClusters)**2) #average of cluster precission for clusters with more or equal
		# number of elements than Nelemns
		temp2 +=temp
	s = sum([1/float(Nelem) for Nelem in eb])

	total = temp2/float(s)
	return total

def PCindex(clustering, dicValues,maxFeats,allFeats):
	'''
	Pick the number of classes according to the contrast value of the features that define each class.
	The contrast value is how good is the f measure of this feature comared to intracluster inertia
	(higher, better). Better for less dimensions
	:param clustering: clustering system
	:param file1: csv file
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:param flag: string, 'max' or ''
	:return: float
	'''

	k = len(clustering)

	temp2 = 0
	for cluster in clustering:
		if cluster in maxFeats:
			contrastSum = 0 #contrast value of a feature for a cluster
			for feat in maxFeats[cluster]:
				c = contrast(clustering, dicValues, feat, cluster,allFeats)
				if c:
					contrastSum += c

			#constrast value of all feats of c divided by number of verbs??
			temp = contrastSum/float(len(clustering[cluster]))

			temp2 += temp #sum for all clustering
	return temp2/float(k)
		

def PCindex2(clustering, dicValues, maxFeats, allFeats):
	'''
	Select number of classes based on the contrast value of the features that define each class. Lamirel modified
	Problem: does not penalize low f measures
	Since it is not normalized by the num of feats of a cluster, clusters with a lot of low feats
	score the same as clusters with clusters with few high feats
	:param clustering: clustering system
	:param file1: csv file
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:param flag: 'max' or ''
	:return: float value
	'''

	k = len(clustering) #number of groups
	temp2 = 0
	for cluster in clustering:
		if cluster in maxFeats:
			contrastSum = 0 #contrast value of features of cluster
			for feat in maxFeats[cluster]:
				c = contrast(clustering, dicValues, feat, cluster, allFeats)
				if c:
					#print 'control', feat, cluster, c
					contrastSum += c
			#print 'cluster', cluster, 'val', contrastSum
			temp2 += contrastSum #sum for all clusering
	return temp2/float(k) #contrast value for all feats and all clusters divided by number of clusters

def indiceQ(clustering, dicValues):
	'''
	minimizes micro-macrorecall
	FMacro larger for smaller values of k and decreases monotonically.
	Fmicro smaller for lower values of k and increases monotonically.
	But if the two are bad, Q index is good
	:param clustering: clustering system
	:param dicValues: dict with info about weight of each feat for each verb{verb:{feat:wight, feat:weight}, verb:{...}}
	:param file1: csv file
	:return: float
	'''

	micro_rec=MicroRecall(clustering, dicValues)
	micro_prec=MicroPrecision(clustering, dicValues)
	FMicro = FmeasureMicro(micro_prec, micro_rec)
	#print 'm',FMicro
	macro_rec=MacroRecall(clustering, dicValues)
	macro_prec=MacroPrecision(clustering, dicValues)
	FMacro = FmeasureMacro(macro_prec, macro_rec)
	#print 'M' ,FMacro
	#print 'r1', FMacro-FMicro
	r=1-abs(FMacro-FMicro)
	return r



def IntraClusterInertia(dataMatrix, clustering, dist):
	'''
	Calculate distances of the datapoints to the medoid of the cluster, better if low
	:param dataMatrix: panda dataframe
	:param clustering: clustering system
	:param dist: distance
	:return: float
	'''

	dDef=0
	for cluster in clustering:
		distC=0
		verbs = clustering[cluster]
		m=[]
		for v in verbs:
			vector = dataMatrix.loc[v]
			floated = [float(e) for e in vector]
			m.append(floated)
		M=numpy.array(m) #matrix with all members of cluster
		medoid = numpy.mean(M,0) ##pasar a int para dice
		#print 'array,med',M, medoid


		if dist == dice:
			temp= [numpy.int64(e) for e in medoid]
			#print '?',temp
			medoid = numpy.array(temp)
			#print 'extramedoid', medoid
		for verb in M:
			if dist == dice:
				Vtemp = [numpy.int64(e) for e in verb]
				distC += dist(Vtemp, medoid)

			else:
				distC+=dist(verb,medoid)

		d2= distC/float(len(verbs))
		dDef += d2
	return dDef/float(len(clustering))
		
		
		
	
def InterClusterInertia(dataMatrix, clustering, dist):
	'''
	calculates distances btw reference points representing medoids of different clusters
	better if high
	:param dataMatrix: pandas df
	:param clustering: clustering system
	:param dist: distance
	:return: float
	'''

	dic= []
	dDef=0
	for cluster in clustering:
		verbs = clustering[cluster]
		m=[]
		for v in verbs:
			vector = dataMatrix.loc[v]
			m.append(vector)
		M=numpy.array(m)
		medoid = numpy.mean(M,0)
		dic.append((cluster, medoid))
	for d in dic:
		for e in dic:
			#print type(d[1]), type(e[1])
			if e[0] !=d[0]: #cluster
				###take into account type of data and metric
				f = dist(e[1],d[1]) #distance between medoids
				dDef += f
				#print d[0],e[0],f
	divisor = float(len(clustering)**2-len(clustering)) #number of comparisons
	#print 'div',divisor, dDef
	return dDef/divisor

