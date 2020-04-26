# -*- coding: utf-8 -*-
import codecs
import os, re, pprint
from collections import OrderedDict
from collections import defaultdict
from frozendict import frozendict
import argparse


sem = ['sumo','tco','periodico', 'supersense', 'lema']
syn = ['morfo','syntax','syntaxSP','morfSPref']
conf = ['patrones','constituyentes','rasgos']
num = ['raw.csv','bin.csv','probabilities.csv']

RolesValidos = ['RA_pat_bin.csv','RC_pat_bin.csv','RM_pat_bin.csv']


def tree(): return defaultdict(tree)

def OrganizeInfo(task,val):
	'''
	:param val result to be compared. Values: 
	'''
	finaldic = tree()
	listadics = set()

	f = codecs.open(task,'r', encoding='utf-8')
	for line in f:
		line.strip('\n')
		lista = line.split(',')
		if 'file' in lista[0]  or 'deps_' in lista[0]: #header
			pass
			
		else:
			tipoRoles = lista[2]
			if tipoRoles in RolesValidos:
				dic = {}
				linF = re.split('[_\+]', lista[0]) #valores de las feats ling

				##aspecto
				if 'aspect' in linF:
					dic['asp'] = 'asp'
				else:
					dic['asp'] = 'noasp'
				
				##others
				for e in linF:
					if e in syn:
						#print e
						if e == 'syntaxSP':
							dic['syn']='syntax'
							#print 'convert 2 syntax'
						if e == 'morfSPref':
							dic['syn']='morfo'
							#print 'convert 2 morfo'
						if e == 'morfo' or e == 'syntax':
							dic['syn']=e
							#print 'add directly'
						
					if e in conf:
						#print e, 'conf'
						dic['conf']=e
						
					if e in num:
						if e == 'bin.csv':
							dic['num']=e
						else:
							dic['num'] = u'nobin'	
									
					if e in sem:
						dic['sem'] = e

				if not 'sem' in dic:
					dic['sem'] = 'nosem'

				numClases = lista[1].split('_')[1]
				linkage = lista[1].split('_')[2]
				metrica = lista[1].split('_')[3]
				
				dic['numC'] = numClases
				dic['linkage'] = linkage
				dic['metrica'] = metrica
				dic['tipoRoles'] = tipoRoles
				
				if val == 'threshold': #turmo
					recall = lista[12]
					prec = lista[13]
					f1 = lista[14]

				if val == 'psico': #psico
					recall = lista[22]
					prec = lista[23]
					f1 = lista[24].strip('\n')

				if val == 'constructions': # const
					recall = lista[9]
					prec = lista[9]
					f1 = lista[9].strip('\n')

				if val == 'sensemClases': #golds
					recall = lista[3]
					prec = lista[3]
					f1 = lista[3].strip('\n')

				#add to tree				
				dic2 = OrderedDict(sorted(dic.items(), key=lambda t: t[0]))
				listadics.add(frozendict(dic2))
				finaldic[dic2['asp']][dic2['conf']][dic2['linkage']][dic2['metrica']][dic2['num']][dic2['numC']][dic2['sem']][dic2['syn']][dic2['tipoRoles']]={'recall':recall,'prec':prec,'f1':f1}

		
	return finaldic, listadics


def getConfigBranchInfo(finaldic, feats, variable):
	return finaldic[feats[0]][variable][feats[2]][feats[3]][feats[4]][feats[5]][feats[6]][feats[7]][feats[8]]

def getAspectBranchInfo(finaldic, feats, variable):
	return finaldic[variable][feats[1]][feats[2]][feats[3]][feats[4]][feats[5]][feats[6]][feats[7]][feats[8]]

def getSemanticBranchInfo(finaldic, feats, variable):
	return finaldic[feats[0]][feats[1]][feats[2]][feats[3]][feats[4]][feats[5]][variable][feats[7]][feats[8]]

def getSyntacticBranchInfo(finaldic, feats, variable):
	return  finaldic[feats[0]][feats[1]][feats[2]][feats[3]][feats[4]][feats[5]][feats[6]][variable][feats[8]]


def outputConfiguration(out, name, listadics, finaldic):
	docuP = open(os.path.join(out, name+'_configuration.csv'), 'w')
	docuP.write('cons,pat\n')
	for element in listadics:
		if element['conf'] == 'patrones':
			feats = [element[f] for f in sorted(element.keys())]
			constituents = getConfigBranchInfo(finaldic, feats, 'constituyentes')
			patterns = getConfigBranchInfo(finaldic, feats, 'patrones')
			if not constituents == {}:
				consScore = constituents['f1']
				patScore = patterns['f1']
				docuP.write(str(consScore) + ',' + str(patScore) + '\n')


def outputAspect(out, name, listadics, finaldic):
	docuP = open(os.path.join(out, name+'_ aspect.csv'), 'w')
	docuP.write('asp,noasp\n')
	for el in listadics:
		if el['asp'] == 'asp':
			feats = [el[f] for f in sorted(el.keys())]
			asp = getAspectBranchInfo(finaldic, feats, 'asp')
			noasp = getAspectBranchInfo(finaldic, feats, 'noasp')
			if not noasp == {}:  #
				aspScore = asp['f1']
				noaspScore = noasp['f1']
				docuP.write(str(aspScore) + ',' + str(noaspScore) + '\n')


def outputSemantics(out, name, listadics, finaldic):
	docuP = open(os.path.join(out, name+'_semantics.csv'), 'w')
	docuP.write('lema,sumo,tco,periodico,supersense,nosem\n')
	for element in listadics:
		if element['sem'] == 'lema':
			feats = [element[f] for f in sorted(element.keys())]
			nosem = getSemanticBranchInfo(finaldic, feats, 'nosem')
			sumo = getSemanticBranchInfo(finaldic, feats, 'sumo')
			tco = getSemanticBranchInfo(finaldic, feats, 'tco')
			nosup = getSemanticBranchInfo(finaldic, feats, 'periodico')
			supersense = getSemanticBranchInfo(finaldic, feats, 'supersense')
			lemma = getSemanticBranchInfo(finaldic, feats, 'lema')

			if nosem != {} and sumo != {} and tco != {} and nosup != {} and supersense != {}:
				lemmaScore = lemma['f1']
				sumoScore = sumo['f1']
				tcoScore = tco['f1']
				noSupScore = nosup['f1']
				supersenseScore = supersense['f1']
				nosemScore = nosem['f1']
				docuP.write(
					str(lemmaScore) + ',' + str(sumoScore) + ',' + str(tcoScore) + ',' + str(noSupScore) + ',' + str(
						supersenseScore) + ',' + str(nosemScore) + '\n')


def outputSytax(out, name, listadics, finaldic):
	docuP = open(os.path.join(out, name+'_syntax.csv'), 'w')
	docuP.write('syntax,morfo\n')
	for el in listadics:

		if el['syn'] == 'morfo':
			feats = [el[f] for f in sorted(el.keys())]
			synFunctions = getSyntacticBranchInfo(finaldic, feats, 'syntax')
			synCategories = getSyntacticBranchInfo(finaldic, feats, 'morfo')

			if not synFunctions == {}:  #
				synFuncScores = synFunctions['f1']
				synCatScores = synCategories['f1']
				docuP.write(str(synFuncScores) + ',' + str(synCatScores) + '\n')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output',  default='../evalresults/comparison/def/',
						help='Output folder for the evaluation files with results')
	parser.add_argument('-i', '--input',  default='../clusterings/2ndBatch/',
						help='Folder that contains the results')

	args = parser.parse_args()

	# gold classes
	finaldic, listadics = OrganizeInfo(os.path.join(args.input, 'goldStandard.csv'), 'sensemClases')
	name = 'goldClasses'
	outputConfiguration(args.output, name ,listadics, finaldic)
	outputAspect(args.output, name, listadics, finaldic)
	outputSemantics(args.output, name, listadics, finaldic)
	outputSytax(args.output, name, listadics, finaldic)

	# extenxibility classes
	finaldic, listadics = OrganizeInfo(os.path.join(args.input, 'extensibility.csv'), 'threshold')
	name = 'extensibility'
	outputConfiguration(args.output, name, listadics, finaldic)
	outputAspect(args.output, name, listadics, finaldic)
	outputSemantics(args.output, name, listadics, finaldic)
	outputSytax(args.output, name, listadics, finaldic)

	# constructions
	finaldic, listadics = OrganizeInfo(os.path.join(args.input, 'constructionsSimilarities.csv'), 'constructions')
	name = 'constructionData'
	outputConfiguration(args.output, name, listadics, finaldic)
	outputAspect(args.output, name, listadics, finaldic)
	outputSemantics(args.output, name, listadics, finaldic)
	outputSytax(args.output, name, listadics, finaldic)

	# psycholinguistic data
	finaldic, listadics = OrganizeInfo(os.path.join(args.input, 'psycholinguisticSimilarities.csv'), 'psico')
	name = 'psycholingData'
	outputConfiguration(args.output, name,listadics, finaldic)
	outputAspect(args.output, name, listadics, finaldic)
	outputSemantics(args.output, name, listadics, finaldic)
	outputSytax(args.output, name, listadics, finaldic)



if __name__ == '__main__':
	main()







