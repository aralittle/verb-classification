# -*- coding: utf-8 -*-
import codecs
import os
dire = '/home/lara/Documents/CODIGO/LastsResults/LastResultsExtensibility/'
output = codecs.open('/home/lara/Documents/CODIGO/LastsResults/task.csv','w',encoding='utf-8')

for fi in os.listdir(dire):
	print fi
	f = codecs.open(dire+fi,'r', encoding='utf-8')
	for line in f:
		#line.strip('\n')
		lista = line.split(',')
		if lista[0] == 'fileResultsEvaluado':
			pass
		else:
			output.write(line)
