#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'lara'

import matplotlib.pyplot as plt
import pandas
import os
import seaborn as sns
import argparse



leg_trans = {'asp':'with aspect' , 'noasp':'without aspect', 'cons':'constituents', 'pat':'subcategorization frames',\
	'rasgos':'independent features','bin':'binary data','nobin':'probabilities','lema': 'lemmas', 'sumo':'SUMO cats.',\
	'tco':'TCO cats.','periodico':'word2vec clusters','supersense':'supersenses','nosem':'not semantic info.',\
	'syntax':'syntactic functions','morfo':'syntactic categories','average':'average', 'complete':'complete','single':'single'}





def plotData(file, output):
	### freq of each F score according to the desired features
	df = pandas.read_csv(file, header=0)
	df2 = df.round(1) #rouns value
	numEls = df.shape[0]
	
	#1: get all existing values
	allvals = set()
	dic = {}

	for i, col in enumerate(df2): #name
		freqs = df2[col].value_counts() #how many cases with a specific frequency
		d = freqs.to_dict() #key: rounded value, val: freq of this val
		for e in d:
			allvals.add(e)

	#2: get values with frequency 0
	for i, col in enumerate(df2): #nombre
		freqs = df2[col].value_counts() #how many cases with a specific frequency
		d = freqs.to_dict() #key: rounded value, val: freq of this val
		for e in allvals: #complete
			if not e in d:
				d[e] = 0
		
		for key, val in d.iteritems(): #normalize (divide freq by the number of data)
			d[key] = val/float(numEls)

		dic[leg_trans[col]]=d


	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_visible(True)
	ax1.spines['left'].set_visible(True)
		

	# Density plot
	for ide, col in enumerate(df2):
		subset = df[col]
		# Draw the density plot
		sns.kdeplot(subset,label = leg_trans[col], linewidth=0.78 ,cumulative=True, ax = ax1)  

	# Plot formatting

	plt.legend(prop={'size': 9}, title = 'Features')
	plt.ylabel('Cumulative probability')
	plt.xlabel('Score')

	# boxplot
	
	sns.set_style('white')
	ax2 = fig.add_subplot(212)

	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.spines['bottom'].set_visible(True)
	ax2.spines['left'].set_visible(True)
	
	df.rename(columns=leg_trans , inplace=True)
	meltedData = pandas.melt(df)
	sns.boxplot(x="variable", y="value", data=meltedData, linewidth=0.5,fliersize=1, ax = ax2)

	plt.xlabel('Features used')
	plt.ylabel('Score')
	plt.xticks(fontsize=8)
	plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)

	name = file.split('.')[0]+'.eps'
	plt.savefig(os.path.join(output, name))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output',  default='../evalresults/comparison/def/',
						help='Output folder for the evaluation files with results')
	parser.add_argument('-i', '--input',  default='../clusterings/2ndBatch/',
						help='Folder that contains the clusterings')

	args = parser.parse_args()
	for file in os.listdir(args.input):
		plotData(file, args.output)



if __name__ == '__main__':
	main()