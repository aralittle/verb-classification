#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas, os, scipy
import matplotlib.pyplot as plt
import numpy as np


folderI = './input/'
folder0 = './output/'
plt.rcParams.update({'font.size': 20})

def autolabel(rects, horAl):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '{0}'.format(float(height)),
                ha=horAl, va='bottom', size = 16 ,visible = True)
                


for fi in os.listdir(folderI):
	print(fi)
	file1 = folderI+fi
	df = pandas.read_csv(file1, index_col=0, header=0)
	data = df.values
	evalData = df.index.values

	#figsize=(30.0, 15.0)
	fig = plt.figure(figsize=(30.0, 15.0))
	fig.subplots_adjust(hspace=.8)
	####fig1
	file1 = folder+'subcategorizations.csv'
	df = pandas.read_csv(file1, index_col=0, header=0)
	data = df.values
	evalData = df.index.values

	ax = fig.add_subplot(211)
	legendData = []
	legendNames = []
	bottom_line = [0]*len(evalData) #base inicial
	colors=['y','b']
	indexC=0
	patch_handles =[]
	acc = []
	for column_name, column in df.transpose().iterrows(): #2
		acc.append([column_name, column])
		
	ind = np.arange(len(column))
	width = 0.65	

	d=ax.bar(ind + 0.25, acc[0][1], 0.5, color='#ffff66', align='center')#, alpha=0.5)
	autolabel(d,'left')
	f=ax.bar(ind, acc[1][1],  0.5, color='#99ff66', align='center')#, alpha=0.5)
	autolabel(f,'right')

	ax.set_xticks(ind + 0.25)
	ax.set_xticklabels(evalData,ha='right',rotation=45)

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)

	namePlot = 'All subcategorization'
	#plt.ylabel('percentage of {0} categories shared'.format(namePlot))
	plt.ylabel('{0} frames shared'.format(namePlot))
	#plt.xlabel('Type of data')
	#plt.title('Percentage of pairs with the same category for {0}'.format(namePlot),  y=1.1)
	plt.title('Percentage of subcategorization frames shared',  y=1.1)
	plt.xlim([-1,ind.size])
	plt.ylim([0,0.4])


	plt.tick_params(
	axis='x',          # changes apply to the x-axis
	which='both',      # both major and minor ticks are affected
	bottom='on',      # ticks along the bottom edge are off
	top='off',         # ticks along the top edge are off
	labelbottom='on')

	plt.legend((d,f), (acc[0][0],acc[1][0]), bbox_to_anchor=(1, 1.15))

	###fig2
	file2 = folder+'low_frequency_subcategorizations.csv'
	df2 = pandas.read_csv(file2, index_col=0, header=0)
	data2 = df2.values
	evalData2 = df2.index.values

	ax2 = fig.add_subplot(212)

	legendData2 = []
	legendNames2 = []
	bottom_line = [0]*len(evalData) #base inicial
	colors=['y','b']
	indexC=0
	patch_handles2 =[]
	acc2 = []
	for column_name2, column2 in df2.transpose().iterrows(): #2
		acc2.append([column_name2, column2])
		
	ind2 = np.arange(len(column))
	width = 0.5	

	d2=ax2.bar(ind2 + 0.25, acc2[0][1], 0.5, color='#ffff66', align='center')#, alpha=0.5)
	#autolabel(d2,'left')
	for x,y in zip(ind2,acc2[0][1]):
		plt.text(x+width/2., y, '%.2f' % y, ha='left', va= 'bottom', size = 16)

	f2=ax2.bar(ind2, acc2[1][1],  0.5, color='#99ff66', align='center')#, alpha=0.5)

	#autolabel(f2,'right')

	for x,y in zip(ind2,acc2[1][1]):
		plt.text(x-0.08/2., y, '%.2f' % y, ha='right', va= 'bottom', size = 16)


	ax2.set_xticks(ind2 + 0.25)
	ax2.set_xticklabels(evalData2,ha='right',rotation=45)

	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.spines['bottom'].set_visible(True)
	ax2.spines['left'].set_visible(True)

	namePlot = 'Low frequency subcategorization'
	#plt.ylabel('percentage of {0} categories shared'.format(namePlot))
	plt.ylabel('{0} frames shared'.format(namePlot))
	plt.xlabel('Type of data')
	#plt.title('Percentage of pairs with the same category for {0}'.format(namePlot),  y=1.1)
	plt.xlim([-1,ind.size])
	plt.ylim([0,0.4])

	plt.tick_params(
	axis='x',          # changes apply to the x-axis
	which='both',      # both major and minor ticks are affected
	bottom='on',      # ticks along the bottom edge are off
	top='off',         # ticks along the top edge are off
	labelbottom='on')

	#plt.show()
	plt.savefig(folderO+'{0}.epg'.format('subs'),bbox_inches='tight')
