#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'lara'
import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import operator
import argparse
import csv

d = ["abrir_18", "cerrar_19", "crecer_1", "dormir_1", "escuchar_1", "estar_14", "explicar_1", "gustar_1", "gestionar_1",
	 "montar_2", "morir_1", "parecer_1", "pensar_2", "perseguir_1", "trabajar_1", "valorar_2", "valer_1", "ver_1",
	 "viajar_1", "volver_1"]


def mediaPond(x,y):
	return (2*x*y)/(x+y)


def translateIntoPair(d, string):
	'''
	Given a list of verbs, and a string "sense_sense" a set containing the two senses with IDs
	:param d: list of all senses used
	:param string: string "sense_sense" from psycholing.data
	:return: a set that contains the two senses with their IDs
	'''

	pair = set()
	l = string.split('_')  # l is the list of senses in pair
	for v in l:
		for sense in d:
			if v == sense.split('_')[0]:
				pair.add(sense)
	if len(pair) == 2:
		return pair

	else:
		print('missing verb in pool of verbs')


def buildDicsByCritRel(csvsim, csvdif, d, rel):
	'''
	Creates a dict object with keys as criteria and values as the list of sense pairs that are the most sim/dissim
	according to each criterium.
	:param csvsim: csv file that contains most similar pairs according to each crit.
	:param csvdif: csv file that contains most dissimilar pairs according to each crit.
	:param d: list of senses in the experiment
	:param rel: list containing relevant criteria
	:return: two dicts. Format {crit:[(va,vb),(...)], ...}
	{'rolLiMed_syntax_pat_probabilities.csv': [set(['gestionar_1', 'perseguir_1']),...
	'''
	dicSim = {}
	dicDif = {}
	for row in csvsim:
		sensePair = translateIntoPair(d, row[1])  # row[1] = senses, no ide
		if row[0] in rel:
			if row[0] not in dicSim:  # crit
				dicSim[row[0]] = [sensePair]
			else:
				dicSim[row[0]].append(sensePair)

	for row in csvdif:
		sensePair = translateIntoPair(d, row[1])
		if row[0] in rel:
			if row[0] not in dicDif:
				dicDif[row[0]] = [sensePair]
			else:
				dicDif[row[0]].append(sensePair)
	return dicSim, dicDif


def buildByVote(dicSim, dicDif):
	'''
	Creates a dict object that contains info about how many votes the sense pair has as similar or dissimilar.
	Keys are frozen sets containing senses with ID. Values are
	the number of times this pair has been voted as similar by all criteria.
	:param dicSim: dict {crit:[(sense,sense),(...)], ...} similar pairs
	:param dicDif: dict {crit:[(sense,sense),(...)], ...} dissimilar pairs
	:return: two dicts, same output (sim and dif), ordered lists by sense pairs and votes,
	number of total criteria
	#[((va,vb),2),((va,vb),4)]
	[(frozenset(['abrir_18', 'cerrar_19']), 23),
	'''

	votesSim = {}  # initializing values: frozen set ([vbos: num votes)]
	votesDif = {}

	for e in dicSim:  # criterion
		simpairs = dicSim[e]  # similar pairs according to a crit
		for d in simpairs:  # for each pair
			f = frozenset(d)
			if f in votesSim:
				votesSim[f] += 1
			else:
				votesSim[f] = 1

	for e in dicDif:
		simpairs = dicDif[e]
		for d in simpairs:
			f = frozenset(d)
			if f in votesDif:
				votesDif[f] += 1
			else:
				votesDif[f] = 1

	sorted_x = sorted(votesSim.items(), key=operator.itemgetter(1), reverse=True)
	sorted_y = sorted(votesDif.items(), key=operator.itemgetter(1), reverse=True)
	maxVotes = len(dicSim)
	return sorted_x, sorted_y, maxVotes


def analyze(fileSim, filedif, rel, d):
	'''
    Initializes all the processes, scoring pairs by votes
    :return: dicts by crit,by vote and number of total votes
    {crit:[(va,vb),(...)], ...}, [((va,vb),2),((va,vb),4)], int
    '''
	csvsim = csv.reader(open(fileSim, 'rb'), quotechar='"')
	csvdif = csv.reader(open(filedif, 'rb'), quotechar='"')
	dicSim, dicDif = buildDicsByCritRel(csvsim, csvdif, d, rel)
	sorted_x, sorted_y, maxVotes = buildByVote(dicSim, dicDif)
	return dicSim, dicDif, sorted_x, sorted_y, maxVotes


def EvalAnecdotic(results, outputs, fileS, fileD, ref):

	with open(os.path.join(results, 'anecConstruPondMed.csv'), 'w') as anecdotic:
		anecdotic.write('file,' + 'clusteringData,')
		dicSim, dicDif, sorted_sim, sorted_dif, maxVotes = analyze(fileS, fileD, ref, d) #global votes

		sims=dicSim.keys()
		for fs in sims:
			anecdotic.write(fs+'-sim,')
			anecdotic.write(fs+'-dif,')
			anecdotic.write(fs+'_score,')

		anecdotic.write('total_sim,total_disim,percSim,percDisim,score\n')
		count = 0
		for file in os.listdir(outputs):
			count+=1
			print('evaluating over these features: ', file, count)
			f = open(os.path.join(outputs + file), 'rb')
			clustering_dic = pickle.load(f) # {agglo_Nclusters_method_metric_sh = ({numClster:[verbos], numCter:[vbos]},[1,2,3,2,1]),...}

			for data, clustering in clustering_dic.iteritems():  # info, clustering
				ldata = data.split('_')
				numC = ldata[1]
				if int(numC) in [i for i in range(2, 201,3)]:
					clusterDict = clustering[0]
					anecdotic.write(file+','+data+',')

					#ALL CRITERIA
					labels_real = []
					for i in range(len(sorted_sim)):
						labels_real.append(0)
					for i in range(len(sorted_dif)):
						labels_real.append(1)
					labels_obtained = []
					sim = 0 #weighted similarity score
					percentSimFound = 0 # percentage of similar pairs adequately found


					for fs in sorted_sim: #[(frozenset(['abrir_18', 'cerrar_19']), 23),
						senses = list(fs[0])
						ThisPairInSameCluster = False
						for c in clusterDict:
							if senses[0] in clusterDict[c] and senses[1] in clusterDict[c]:
								ThisPairInSameCluster = True
								break
								#print('both senses in cluster')
						if ThisPairInSameCluster: #votes that the pair gets as similar divided by the number of criteria (maximum of votes)
							val = fs[1]/float(maxVotes) ## similarity is weighted by the amount of votes obtained by the pair
							percentSimFound += 1
							sim +=1*val
							labels_obtained.append(0) #0 is the category for similar
						else:
							labels_obtained.append(1) #1 is the category for dissimilar
					totalSim = sim/float(len(sorted_sim)) #weighted amount of sim pairs found / total amount of sim pairs
					totalSPerc = percentSimFound/float(len(sorted_sim)) #amount of sim pairs found / total amount of sim pairs


					disim = 0
					percentDisimFound = 0
					for e in sorted_dif: #pair
						senses = list(e[0])
						ThisPairInSameCluster = False
						for c in clusterDict:
							if senses[0] in clusterDict[c] and senses[1] in clusterDict[c]:
								ThisPairInSameCluster = True
								break
						if not ThisPairInSameCluster:
							val = e[1]/float(maxVotes) ## desimilarity is weighted by the amount of votes obtained by the pair
							disim +=1*val
							percentDisimFound +=1
							labels_obtained.append(1) #label as dissimilar
						else:
							labels_obtained.append(0) #label as similar
					totalDisim = disim/float(len(sorted_dif)) #weighted amount of disim pairs found / total amount of disim pairs
					totalDPerc = percentDisimFound / float(len(sorted_dif)) #amount of disim pairs found / total amount of disim pairs
					score = mediaPond(totalSPerc, totalDPerc)


					#BY CRITERIUM
					dicScore= {}
					for crit in dicSim:
						dicScore[crit] = ([],[]) #for each crit, keep similar (0) and dissimilar pairs (1)
						v = 0
						for verbPair in dicSim[crit]:
							ThisPairInSameCluster = False
							dicScore[crit][0].append(0)# gold
							senses = list(frozenset(verbPair))
							for c in clusterDict:
								if senses[0] in clusterDict[c] and senses[1] in clusterDict[c]:
									ThisPairInSameCluster = True
									break
							if ThisPairInSameCluster:
								v += 1
								dicScore[crit][1].append(0)# prediction: they are similar
							else:
								dicScore[crit][1].append(1)# prediction: they are not similar

						totalSim2=v/float(len(dicSim[crit])) #humber of similar pairs in same cluster/ number of sim pairs
						anecdotic.write(str(round(totalSim2,3))+',')

						v2 = 0
						for verbPair in dicDif[crit]:
							dicScore[crit][0].append(1)  # 1
							ThisPairInSameCluster = False
							senses = list(frozenset(verbPair))
							for c in clusterDict:
								if senses[0] in clusterDict[c] and senses[1] in clusterDict[c]:
									ThisPairInSameCluster = True
									break
							if not ThisPairInSameCluster:
								v2 += 1
								dicScore[crit][1].append(1)  #2
							else:
								dicScore[crit][1].append(0)  # 2
						totalDif2=v2/float(len(dicDif[crit]))
						anecdotic.write(str(round(totalDif2,3))+',')
						scoreCrit = mediaPond(totalSim2, totalDif2)
						anecdotic.write(str(scoreCrit)+',')
					anecdotic.write(str(round(totalSim,3))+','+str(round(totalDisim,3))+','+str(round(totalSPerc,3))+','+str(round(totalDPerc,3))+','+str(round(score,3))+'\n')



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output',  default='../evalresults/anec/def/',
						help='Output folder for the evaluation files with results')
	parser.add_argument('-i', '--input',  default='../clusterings/2ndBatch/',
						help='Folder that contains the clusterings')
	parser.add_argument('-s', '--filesim',  default='../Auxdata/sim1.csv',
						help='File that contains similarity scores per criterion')
	parser.add_argument('-d', '--filedif',  default='../Auxdata/dif.csv',
						help='File that contains disimilarity scores per criterion')
	parser.add_argument('crit', choices=['constructions','WA'],help='Criterion to use in order to perform the comparison (constructions data or WA data)')

	args = parser.parse_args()

	if args.crit == 'constructions':
		ref = ['teoria_bin.csv']
	else:
		ref = ['lex_allW_0_0_probabilities-3p.csv',
			   'rich_allW_supersense_1S_probabilities-3p.csv',
			   'rich_allW_hyper_1S_probabilities-3p.csv',
			   'rich_allW_sumo_1S_probabilities-3p.csv',
			   'rich_allW_TCOall_1S_probabilities-3p.csv',
			   'rich_allW_TCO_1S_probabilities-3p.csv'
			   ]
	EvalAnecdotic(args.output, args.input, args.filesim, args.filedif, ref)

if __name__ == '__main__':
	main()
