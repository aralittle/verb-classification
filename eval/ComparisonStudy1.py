# -*- coding: utf-8 -*-
'''
Created on 29/06/2016

@author: lara

evaluacion de los clusters on a feature basis
'''

from collections import OrderedDict, Counter
import operator, itertools, pandas, csv, numpy
from scipy.spatial.distance import pdist, cosine, dice
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

path_casa = '/media/lara/OS/Users/Lara/Google Drive'
path_ofi = '/home/user/Google Drive'
path_uso = path_ofi


#fileSim = path_uso + '/phd_projects/proyectos/similitud_verbal/datos/sim1.csv'
#filedif = path_uso + '/phd_projects/proyectos/similitud_verbal/datos/dif.csv'
#d = ["abrir_18","cerrar_19","crecer_1","dormir_1","escuchar_1","estar_14","explicar_1","gustar_1","gestionar_1", "montar_2","morir_1", "parecer_1", "pensar_2", "perseguir_1","trabajar_1","valorar_2","valer_1","ver_1","viajar_1","volver_1"]


##prueba
fileSim = path_uso + '/phd_projects/proyectos/data/csvs/pruebas/psico/sim1.csv'
filedif = path_uso + '/phd_projects/proyectos/data/csvs/pruebas/psico/dif.csv'
d = ["V_1","V_2","V_3","V_4"]




def translateIntoPair(d, string):
	pair = set()
	l=string.split('_') #list of senses in pair
	print l
	for v in l:
		#print 'v',v
		for sense in d:
			#print 's',sense
			if v == sense.split('_')[0]:
				pair.add(sense)
	if len(pair) ==2:
		return pair
	else:
		print 'missing verb in pool of verbs'
				
def buildDicsByCrit(csvsim,csvdif, d):
	dicSim = {}
	dicDif = {}

	for row in csvsim:
		#print '1',row[1]
		#print '0',row[0]
		sensePair =  translateIntoPair(d, row[1]) #row[1] = sentidos, output set (sense, sense) no ide
		if row[0] not in dicSim: #crit		
			dicSim[row[0]] = [sensePair]
		else:
			dicSim[row[0]].append(sensePair)

	for row in csvdif:
		sensePair =  translateIntoPair(d, row[1])
		if row[0] not in dicDif:
			dicDif[row[0]] = [sensePair]
		else:
			dicDif[row[0]].append(sensePair)
	return dicSim, dicDif #{crit:[(va,vb),(...)], crit:[]}		

def buildByVote(dicSim, dicDif):
	'''
	votos globales
	'''
	print 'mv',dicSim
	votesSim = {} #frozen set con vbos: num votos
	votesDif = {}

	for e in dicSim:#criterio (teoria, etc)

		simpairs = dicSim[e] #pares sim segun crit A

		for d in simpairs: #pareja en lista
			f= frozenset(d)
			if f in votesSim:
				votesSim[f]+=1
			else:
				votesSim[f] = 1
	print '---vs',votesSim			
	for e in dicDif:
		simpairs = dicDif[e]
		for d in simpairs:
			f= frozenset(d)
			if f in votesDif:
				votesDif[f]+=1
			else:
				votesDif[f] = 1

	sorted_x = sorted(votesSim.items(), key=operator.itemgetter(1), reverse= True) 
	sorted_y = sorted(votesDif.items(), key=operator.itemgetter(1), reverse= True) 
	maxVotes =  len(dicSim)
	#{crit:[(va,vb),(...)], crit:[]}, [((va,vb),2),((va,vb),4)], num de criterios
	return dicSim, dicDif,sorted_x, sorted_y, maxVotes

	
def main():
	csvsim = csv.reader(open(fileSim, 'rb'), quotechar = '"')
	csvdif = csv.reader(open(filedif, 'rb'), quotechar = '"')
	dicSim, dicDif = buildDicsByCrit(csvsim,csvdif, d)

	dicSim, dicDif,sorted_x, sorted_y, maxVotes = buildByVote(dicSim, dicDif)
	#print sorted_x#, sorted_y
	return dicSim, dicDif, sorted_x, sorted_y, maxVotes
		

#main()

'''
parejas_similares = []#cada pareja una tupla
parejas_disimilares = []

	
#clustering_dic = {0:['v1','v3'], 1:['v2','v4']}

clustering_dic = {0:['v1','v2'], 1:['v3', 'v4']}


origFile = path_uso + '/phd_projects/proyectos/data/csvs/training/borrar/borrar.csv'
dicValues = csvToDic(origFile)
dataMatrix= pandas.read_csv(origFile, index_col=0, header=0)


sim = 0

for e in parejas_similares:
	for c in clustering_dic:
		if e[0] in clustering_dic[c] and e[1] in clustering_dic[c]:
			sim +=1
totalSim = sim/float(len(parejas_similares))


disim = 0
for e in parejas_disimilares:
	for c in clustering_dic:
		if e[0] in clustering_dic[c] and not e[1] in clustering_dic[c]:
			disim +=1
totalDisim = disim/float(len(parejas_disimilares))
			
'''
