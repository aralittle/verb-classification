#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'lara'
import warnings
warnings.filterwarnings("ignore")
import sys
import operator
reload(sys)
sys.setdefaultencoding('utf-8')

# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas
import operator
import os
import seaborn as sns

path_casa = '/media/lara/OS/Users/Lara/Google Drive'
path_ofi = '/home/user/Google Drive'
path_uso = path_ofi


#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/averaged4allroles2/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/abstract2/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/medium2/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/specific2/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/constru/def2/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/psico/def2/'


#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/comparison/comparison_all/'
#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/comparison/abstract/'
#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/comparison/medium/'
#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/comparison/specific/'
#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/cons/'
#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/psico/'

##no rasgos
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/abstract2noRasgos/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/averaged4allRoles2Norasgos/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/medium2noRasgos/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/specific2NoRasgos/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/psico/def2NoRasgos/'
path = '/home/lara/Documents/CODIGO/comparacionFeats/data/constru/def2noRasgos/'


#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/comparisonNoRasgos/abstractNoRasgos/'
#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/comparisonNoRasgos/comparisonAllNoRasgos/'
#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/comparisonNoRasgos/mediumNoRasgos/'
#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/comparisonNoRasgos/specificNoRasgos/'
#output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/psicoNoRasgos/'
output = '/home/lara/Documents/CODIGO/comparacionFeats/plots/2/consNoRasgos/'



leg_trans = {'asp':'with aspect' , 'noasp':'without aspect', 'cons':'constituents', 'pat':'subcategorization frames',\
	'rasgos':'independent features','bin':'binary data','nobin':'probabilities','lema': 'lemmas', 'sumo':'SUMO cats.',\
	'tco':'TCO cats.','periodico':'word2vec clusters','supersense':'supersenses','nosem':'not semantic info.',\
	'syntax':'syntactic functions','morfo':'syntactic categories','average':'average', 'complete':'complete','single':'single'}



for file1 in os.listdir(path):
	### freq of each F score according to the desired features

	
	df = pandas.read_csv(path+file1, header=0)
	df2 = df.round(1) #rouns value
	numEls = df.shape[0]
	print numEls
	
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
		print col
		freqs = df2[col].value_counts() #how many cases with a specific frequency
		d = freqs.to_dict() #key: rounded value, val: freq of this val
		
		for e in allvals: #complete
			if not e in d:
				d[e] = 0
		
		for key, val in d.iteritems(): #normalize (divide freq by the number of data)
			d[key] = val/float(numEls)

		dic[leg_trans[col]]=d

		
	
	dfDef = pandas.DataFrame(data=dic)


	fig = plt.figure()
	params = {'legend.fontsize': 8,
          'legend.handlelength': 2}
	plt.rcParams.update(params)
	

	#option 1
	'''
	ax1 = fig.add_subplot(111)
	binNum = 6
	mini = min(dfDef.min().values)
	maxi = max(dfDef.max().values)
	ticks = np.linspace(mini,maxi,num=binNum)	
	plt.hist(dfDef.values, bins=ticks, align='left', label=list(dfDef))
	plt.xticks(ticks.round(2))
	plt.legend()
	'''
	###
	
	#opcion 2
	ax1=dfDef.plot(kind='bar')
	###

	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_visible(True)
	ax1.spines['left'].set_visible(True)
	#ax1.set_xlabel("F-measure (rounded)")
	ax1.set_xlabel("Score (rounded)")
	ax1.set_ylabel("Probability")
	
	#plt.show()
	

	
	name = file1.split('.')[0]+'.eps'

	plt.savefig(output+'BAR'+name)

'''
binned_data = np.array([[41., 3., 3.], [ 8., 3., 3.], [ 1., 2., 2.], [ 2., 7., 3.],
                        [ 0., 20., 0.], [ 1., 21., 1.], [ 1., 2., 4.], [ 3., 4., 23.],
                        [ 0., 0., 9.], [ 3., 1., 14.]]).T

# The shape of the data array have to be:
#  (number of categories x number of bins)
print(binned_data.shape)  # >> (3, 10)

x_positions = np.array([0.1, 0.34, 0.58, 0.82, 1.06, 1.3, 1.54, 1.78, 2.02, 2.26])

number_of_groups = binned_data.shape[0]
fill_factor =  .8  # ratio of the groups width
                   # relatively to the available space between ticks
bar_width = np.diff(x_positions).min()/number_of_groups * fill_factor

colors = ['red','yellow', 'blue']
labels = ['red flowers', 'yellow flowers', 'blue flowers']

for i, groupdata in enumerate(binned_data): 
    bar_positions = x_positions - number_of_groups*bar_width/2 + (i + 0.5)*bar_width
    plt.bar(bar_positions, groupdata, bar_width,
            align='center',
            linewidth=1, edgecolor='k',
            color=colors[i], alpha=0.7,
            label=labels[i])

plt.xticks(x_positions);
plt.legend(); plt.xlabel('flower length'); plt.ylabel('count');
'''
