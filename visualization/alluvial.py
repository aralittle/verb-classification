#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pickle
import argparse
import plotly, random
import plotly.plotly as py

plotly.tools.set_credentials_file(username='lgilva', api_key='key')



def transformSensemData(fileGoldClasses, clusterDict):
	## clases sensem
	abstractionLevels = ['sensem', 'medium', 'abstract']
	dictsClases = []

	with open(fileGoldClasses, 'r') as docClases:
		for claseRow in range(len(abstractionLevels)):
			classificationTime = {}
			for line in docClases:
				lista = line.split(',')
				classificationTime[lista[0]] = set([lista[claseRow + 1].strip('\n\r')])
			docClases.seek(0)
			dictsClases.append(classificationTime)

		ClasesDic = dictsClases[2]
		setTotal = set()
		clasesCat = []
		labels = [] ##verbos
		for verb, setClas in ClasesDic.iteritems():
			el = random.sample(setClas, 1) #lista
			setTotal.add(el[0])
			clasesCat.append(el[0])
			labels.append(verb)

		mapinClass =  {val:ind for ind, val in enumerate(setTotal)} # label, 1
		mapinClassInv =  {ind:val for ind, val in enumerate(setTotal)} # 1, label
		clasNum = [mapinClass[c] for c in clasesCat] ##clases con mapping a numeros


		labelsSensem = []
		for v in labels:
			for clu in clusterDict:
				if v in clusterDict[clu]:
					labelsSensem.append(clu)
		s = set(labelsSensem)

		etiqCorrel = [] ###etiquetas sensem
		for e in labelsSensem:
			equiv = e+len(set(clasNum))-1
			etiqCorrel.append(equiv)

		#pesos
		di = {}
		for ind, v in enumerate(labels):
			claseSen = etiqCorrel[ind]
			claseAn = clasNum[ind]

			if claseSen in di:
				if claseAn in di[claseSen]:
					di[claseSen][claseAn] +=1
				else:
					di[claseSen][claseAn] = 1

			else:
				di[claseSen] = {claseAn:1}
	return di, mapinClassInv, clasNum, labelsSensem
			
def plotAlluvial(di, mapinClassInv, clasNum, labelsSensem):
	sourceS = []
	targetT = []
	valueV = []
	for s, targs in di.iteritems():
		for t in targs:
			sourceS.append(s)
			targetT.append(t)
			valueV.append(targs[t])

	labelDia = [mapinClassInv[i] for i in set(clasNum)] + ['class '+str(f) for f in set(labelsSensem)]

	data = dict(
		type='sankey',
		orientation = 'h',
		node = dict(
		  pad = 15,
		  thickness = 20,
		  line = dict(
			color = "black",
			width = 0.5
		  ),
		  label = labelDia,
		  color = ["blue" for e in range(len(set(clasNum))) ] + ["red" for e in range(len(set(labelsSensem))) ]
		),
		link = dict(
		  source = sourceS, #si van al mismo o no
		  target = targetT,
		  value = valueV
	  ))

	layout =  dict(
		#title = "Automatic classes vs gold classes",
		font = dict(
		  size = 8
		)
	)


	fig = dict(data=[data], layout=layout)
	py.iplot(fig, validate=False)
		
			
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='Folder that contains the clusterings')
	parser.add_argument('-c', '--classes',  help='Folder that contains the gold classes')
	args = parser.parse_args()


	docClu = open(args.input, 'r')
	clusterElement = pickle.load(docClu)
	for data, clustering in clusterElement.iteritems():  # info, clustering

		if data == 'agglo_14_average_dice_0.255':  # best abstract f1
			clusterDict = clustering[0]  # clase:verbo
			di, mapinClassInv, clasNum, labelsSensem = transformSensemData(args.classes, clusterDict)
			plotAlluvial(di, mapinClassInv, clasNum, labelsSensem)
			break


if __name__ == '__main__':
	main()