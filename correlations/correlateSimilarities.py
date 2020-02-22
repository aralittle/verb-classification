#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas, os, scipy
import argparse
from scipy.spatial.distance import pdist, cosine, dice
from scipy.stats import spearmanr, linregress
from operator import itemgetter
from collections import Counter



def vectorizeDic(dic):
	'''
	Puts the values in a list
	:param dic:
	:return: list containing values
	'''
	v=[]
	for verbP in dic:
		val=verbP[1]
		v.append(val)
	return v


def DistanceAmongVerbs(file1, distance):
	'''
	Measure and save pairwise distances
	:param file1: file containing verb characterizations
	:param distance: distance to be used (cosine or dice)
	:return: dic containing pairwise distances {(verb verb): similarity score}
	'''
	#load in dataframe
	df=pandas.read_csv(file1, index_col=0, header=0)
	data = df.values
	labels = df.index.values #verbs
	#for each row, compute similarity with all the others
	indexv = 0
	indexj = 0

	dict1 = Counter() #ordenado de mayor a menor distancia
	for v in data:

		label1 = labels[indexv]
		for j in data:
			label2 = labels[indexj]
			
			#if they are two different verbs, measure distance
			if label1 != label2:
				labelGeneral = frozenset({label1.split('_')[0],label2.split('_')[0]}) #keep lemma
				if not labelGeneral in dict1:
					#print v,j
					dict1[labelGeneral]=distance(v,j)
			#pass to next row
			if indexj == len(labels)-1:
				indexj = 0
			else:
				indexj+=1   

		#if we are finished, update v
		if indexv == len(labels)-1:
			indexv = 0

		else:
			indexv+=1
	return dict1



def simCalc(doc1,distance):
	'''

	:param doc1: file de teoria
	:return: dict de dict, valores vectorizados
	'''
	teo = DistanceAmongVerbs(doc1, distance)
	neo=dict()
	
	for verb in teo:#frozenset
		v='_'.join(sorted(verb))#conver set in aphabetically ordered string
		value = teo[verb]
		neo[v]=value
	ordered = sorted(neo.items(), key=itemgetter(0)) #ordena alfabeticamnte
	return neo, ordered #dict, alphabetically ordered dict




def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-wa', '--wordAssociations',  default='./DistanceData/WA_3pAllw/',
						help='Folder containing WA distances')
	parser.add_argument('-sr', '--corpus',  default='./DistanceData/Corpus/',
						help='Folder that contains Corpus distances')
	parser.add_argument('-co', '--constructions', default='./DistanceData/constructions/',
                        help='Folder that contains construction distances')
	parser.add_argument('-r', '--results', default='./results/', help='Folder where results are saved')
	parser.add_argument('selection', nargs='*', choices =['wa','constr', 'corpus'] ,help='Two perspectives to be correlated (Word Associations (wa), Constructions (constr), Corpus (corpus))')

	args = parser.parse_args()


	if len(args.selection) < 2:
		print('Select two perspectives from the following: wa (Word Associations), const (Constructions), corpus (Corpus)')
	else:
		relevant = set()
		if 'wa' in args.selection:
			dicWAs = {}
			for WAfile in os.listdir(args.wordAssociations):
				if 'probabilities' in WAfile:
					info = WAfile.split('_')
					type = info[2]
					dicWA = simCalc(args.wordAssociations + WAfile, cosine)[1]
					vectorAsoc = vectorizeDic(dicWA)
					dicWAs[type] = vectorAsoc
			relevant.add('wa')

		if 'constr' in args.selection:
			for teo in os.listdir(args.constructions):
				dicTeo = simCalc(args.constructions + teo, dice)[1]  # alphabetically ordered
				vectorTeo = vectorizeDic(dicTeo)  # vector with distances
			relevant.add('constr')

		if 'corpus' in args.selection:
			dicCorp = {} # dic {rolLI_syn_cons:[0,0.1,03...]}
			for COfile in os.listdir(args.corpus):
				dicCO = simCalc(args.corpus + COfile, cosine)[1]
				vectorCO = vectorizeDic(dicCO)
				dicCorp[COfile] = vectorCO
			relevant.add('corpus')


		if relevant == set(['corpus', 'constr']):
			results = open(args.results + 'TeoRolesSpearman.csv', 'wb')
			for tipoCorpus in dicCorp:
				corr2, r2 = spearmanr(vectorTeo, dicCorp[tipoCorpus])
				if r2 < 0.05:
					results.write(tipoCorpus + ',')
					results.write(str(corr2) + '\n')


		if relevant == set(['corpus', 'wa']):
			results = open(args.results + 'RolWA.csv', 'wb')
			WAs = dicWAs.keys()
			results.write(' ,' + ','.join(WAs))
			results.write('\n')

			TiposCo = dicCorp.keys()
			for tipoCorpus in TiposCo:
				results.write(tipoCorpus + ',')
				for tipoWA in WAs:
					corr2, r2 = spearmanr(dicCorp[tipoCorpus], dicWAs[tipoWA])
					if r2 < 0.05:
						results.write(str(corr2) + ',')
					else:
						results.write('NS' + ',')
				results.write('\n')

		if relevant == set(['wa', 'constr']):
			results = open(args.results + 'TeoWA.csv', 'wb')
			for tipoWA in dicWAs:
				corr2, r2 = spearmanr(vectorTeo, dicWAs[tipoWA])
				if r2 < 0.05:
					results.write(tipoWA + ',')
					results.write(str(corr2) + '\n')


if __name__ == "__main__":
	main()

