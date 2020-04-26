# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import codecs,pandas
import os, re, pprint
from collections import OrderedDict
from collections import defaultdict
from scipy import stats

path_ofi = '/home/user/Google Drive'
path_casa = '/media/lara/OS/Users/Lara/Google Drive'
path_uso = path_casa

paths = [
'/home/lara/Documents/CODIGO/comparacionFeats/data/constru/def2noRasgos/',
'/home/lara/Documents/CODIGO/comparacionFeats/data/psico/def2NoRasgos/',
'/home/lara/Documents/CODIGO/comparacionFeats/data/tareaLast/',
'/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/abstract2noRasgos/'
#'/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/averaged4allroles2/',
#'/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/medium2/',
#'/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/specific2/'
]

#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/abstract2noRasgos/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/averaged4allRoles2Norasgos/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/medium2noRasgos/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/sensemDEF/specific2NoRasgos/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/psico/def2NoRasgos/'
#path = '/home/lara/Documents/CODIGO/comparacionFeats/data/tareaLast/'

#path_uso + '/phd_projects/resultadosGlobales/ancora/'
#path_uso + '/phd_projects/resultadosGlobales/datatask/'
#path_uso + '/phd_projects/resultadosGlobales/out/'
#path_uso + '/phd_projects/resultadosGlobales/psico/'
#]
#paths = []

for folder in paths:
	for file1 in os.listdir(folder):
		print '...',folder.split('/')[-3:]
		
		df = pandas.read_csv(folder+file1, header=0) #index_col=0
		#print df.describe()
		#for col in df:
			#print col
			#print df[col].mean()

		if file1 == 'aspect.csv':
			F, p = stats.f_oneway(df['asp'], df['noasp'])
			print F,p,'asp'
			F, p = stats.kruskal(df['asp'], df['noasp'])
			print F,p,'asp np'		

		
		if file1 == 'configuration.csv':	
			F, p = stats.f_oneway(df['cons'], df['pat'])
			print F, p, 'config'
			F, p = stats.kruskal(df['cons'], df['pat'])
			print F, p, 'config np'		
				
		if file1 == 'counts.csv':	
			F, p = stats.f_oneway(df['nobin'], df['bin'])
			print F,p, 'bins'
			F, p = stats.kruskal(df['nobin'], df['bin'])
			print F,p, 'bins np'
		
		if file1 == 'syntax.csv':
			F, p = stats.f_oneway(df['syntax'], df['morfo'])
			print F, p, 'syntax'
			F, p = stats.kruskal(df['syntax'], df['morfo'])
			print F, p, 'syntax np'
		
		if file1 == 'linkage.csv':
			F, p = stats.f_oneway(df['average'], df['complete'],df['single'])
			print F,p, 'linkage'
			F, p = stats.kruskal(df['average'], df['complete'],df['single'])
			print F,p, 'linkage np'
			
		if file1 == 'semantics.csv':
			F, p = stats.f_oneway(df['lema'], df['sumo'],df['tco'],df['supersense'], df['periodico'],df['nosem'])
			print F, p, 'semantica'
			F, p = stats.kruskal(df['lema'], df['sumo'],df['tco'],df['supersense'], df['periodico'],df['nosem'])
			print F, p, 'semantica np'


'''
v1 = '/home/lara/Documents/CODIGO/clustering/evalresults/comparison/def/comparacionSensemClassesF1abstract.BORRAR.csv'
v2 = '/home/lara/Documents/CODIGO/clustering/evalresults/comparison/def/comparacionSensemClassesF1abstract.csv'
v3 = '/home/lara/Documents/CODIGO/clustering/evalresults/comparison/def/comparacionSensemClassesF1medium.BORRAR.csv'
v4 = '/home/lara/Documents/CODIGO/clustering/evalresults/comparison/def/comparacionSensemClassesF1medium.csv'
v5 = '/home/lara/Documents/CODIGO/clustering/evalresults/comparison/def/comparacionSensemClassesF1sensem.BORRAR.csv'
v6 = '/home/lara/Documents/CODIGO/clustering/evalresults/comparison/def/comparacionSensemClassesF1sensem.csv'


from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
		
df1 = pandas.read_csv(v1, header=0, index_col=0)
df2 = pandas.read_csv(v2, header=0, index_col=0)

print pearsonr(df1['v'],df2['F1'])
plt.scatter(df1['v'],df2['F1'])
plt.show()


df1 = pandas.read_csv(v3, header=0, index_col=0)
df2 = pandas.read_csv(v4, header=0, index_col=0)

print pearsonr(df1['v'],df2['F1'])
plt.scatter(df1['v'],df2['F1'])
plt.show()

df1 = pandas.read_csv(v5, header=0, index_col=0)
df2 = pandas.read_csv(v6, header=0, index_col=0)

print pearsonr(df1['v'],df2['F1'])
plt.scatter(df1['v'],df2['F1'])
plt.show()
'''
