# -*- coding: utf-8 -*-
'''
Created on 23/07/2014
@author: lara
'''
import argparse
from lxml import etree as ET
from collections import OrderedDict, Counter
from sklearn import preprocessing
import operator
import itertools
import numpy as np
import pickle
import os
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


prep =  ['a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en' , 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'según'.decode('utf-8'), 'sin', 'so', 'sobre', 'tras', 'versus ', 'vía', 'vía'.decode('utf-8')]



roles_mapping = {'Ag/caus': 'Actor-Actor-Actor', 'Goal_ag': 'Actor-Agent-Agent', 'Ag_exp': 'Actor-Agent-Agent',
		   'Ag_source': 'Actor-Agent-Agent', 'Mov-ag_T': 'Actor-Agent-Agent', 'Ag_source(pl)': 'Actor-Agent-Agent',
		   'Ag(pl)': 'Actor-Agent-Agent',
		   'Agent': 'Actor-Agent-Agent', 'Measure': 'Undergoer-Attrib-Attrib', 'Ind-caus': 'Actor-Cause-Cause',
		   'Caus': 'Actor-Cause-Cause', 'Circ': 'Circ', 'Comitative': 'Undergoer-Theme-Theme',
		   'Qual': 'Undergoer-Attrib-Attrib', 'Goal': 'Place-Goal-Goal',
		   'Exp': 'Undergoer-Patient-Experiencer', 'Af-creat-T': 'Undergoer-Patient-Patient',
		   'Exp(pl)': 'Undergoer-Patient-Experiencer', 'Purpose': 'Place-Goal-Goal', 'Inic': 'Actor-Actor-Actor',
		   'Inic(pl)': 'Actor-Actor-Actor', 'Instr': 'Undergoer-Instrument-Instrument',
		   'Loc': 'Place-Location-Location', 'Ma': 'Undergoer-Attrib-Attrib',
		   'Means': 'Undergoer-Instrument-Instrument', 'Source': 'Place-Source-Source',
		   'Perc': 'Undergoer-Patient-Experiencer', 'Path': 'Place-Place-Place', 'Subst': 'Undergoer-Theme-Theme',
		   'Af-destr-T': 'Undergoer-Patient-Patient', 'Af-vict-T': 'Undergoer-Patient-Patient',
		   'Af-T': 'Undergoer-Patient-Patient', 'Af-T(pl)': 'Undergoer-Patient-Patient',
		   'Mov-T': 'Undergoer-Theme-Theme', 'T-is': 'Place-Source-Source', 'T-rs': 'Place-Location-Location',
		   'T': 'Undergoer-Theme-Theme', 'T(pl)': 'Undergoer-Theme-Theme', 'Time-to': 'Time-Final_time-Final_time',
		   'Time-at': 'Time-Time-Time', 'Time-from': 'Time-Init_time-Init_time'}


MorfoDic={'Adj':['AdjP','PartSC'],'Adv':['AdvSC','NegAdvP','AdvP','GerSC'],'Pron':['sigla-Pr-Int','PersPron','RelSC','PronP'],\
'PP':['PP-Pron','PP-RelPron','PP','PP-AdvSC','PP-RelSC','PP-CondSC','PP-InfSC','PP-IntSC','PP-Comp','PP-PersPron',\
'PP-ComplSC','sigla-SP-Pr-Int', ],'NP':['NP','InfSC','Proper.Noun'],'Sent':['CondSC','CompP','IntSC','ComplSC','RelPron','RedSC','DirSpSC']}

class argument():
	def __init__(self):
		self.abstractRole = ''
		self.mediumRole = ''
		self.sensemRole = ''
		self.synFunct = ''
		self.synFunctSelectPref = ''
		self.synCat = ''
		self.synCatPrep = ''
		self.synCatCluster = ''
		self.synCatPrepCluster = ''
		self.ontoTCO = []
		self.ontoTCOSplit = []
		self.ontoSupersense = []
		self.ontoSupersense = []
		self.ontoSumo = []
		self.lemma = []

	def get_roles(self, argument):
		syntax = a.attrib["fs"] + '_SS'



class corpus():
	def __init__(self, corpus_file):
		self.corpus_file = corpus_file
		self.sentence_list = []
		self.sentencedic = {}

	def get_all_sentences(self):
		'''
		Parses xml file and selects all nodes
		:param doc: string, xml doc from which info is gonna be extracted
		:return: ist of nodes that represent sentences
		'''

		parser = ET.XMLParser(encoding = 'utf-8')
		root = ET.parse(self.corpus_file, parser)
		sentences = root.iter('sentence')
		for s in sentences:
			self.sentence_list.append(s)

	def get_sentenceLevel_info(self):
		for s in self.sentences:

		#prepare for level
		RolesCon = []
		RolesMed = []
		RolesAbs = []

		#sentence info
		infoFraseL=[]

		#sentence level info
		sense = s.find(".//lexical").attrib["sense"]
		vlema = s.find(".//lexical").attrib['verb']
		ide=s.attrib['id']
		#print ide
		perifrastico = s.find(".//lexical[@periphrastic]")

		if 'perif' in tipo_info:
			infoFraseL.append('p_'+perifrastico.attrib['periphrastic'])


		if 'aspectuality' in s.find(".//semantics").attrib:
			aspectualidad = s.find(".//semantics").attrib['aspectuality']
			if 'aspectual' in tipo_info:
				infoFraseL.append(aspectualidad)


		if 'modality' in s.find(".//semantics").attrib:
			modality = s.find(".//semantics").attrib['modality']
			if 'modal' in tipo_info:
				infoFraseL.append(modality)



		if 'polarity' in s.find(".//semantics").attrib:
			polarity = s.find(".//semantics").attrib['polarity']
			if 'polar' in tipo_info:
				infoFraseL.append(polarity)


		aspect1 = s.find(".//argumental").attrib['aspect']
		aspect = 'stative' if aspect1 == 'State' else 'dynamic'

		if 'aspect' in tipo_info:
			infoFraseL.append(aspect)

		if 'construction' in s.find(".//argumental").attrib:
			cons1=[]
			constr = s.find(".//argumental").attrib['construction']
			if constr != None:
				constS = constr.split('-')
				for cc in constS:
					if cc != '' and cc != ' ':
						cc=cc.strip()
						cons1.append(cc)
			c='*'.join(cons1)
			if 'const' in tipo_info:
				infoFraseL.append(c)

def removeLowFrequentFeats(dicOfAllFeatsPerVerb,Minfreq):
	'''
	Removes elements athat have frequencies that are below a threshold

	:param dicOfAllFeatsPerVerb: dict that contains all the feats and their freqs
	:param Minfreq: int, threshold
	:return: dict with reduced feats, set of feats
	'''

	counter = Counter()
	for verb, val in dicOfAllFeatsPerVerb.iteritems():
		for subcat, freq in val.iteritems():
			if subcat in counter:
				counter[subcat] += freq
			else:
				counter[subcat] = freq

	c2={k:v for k, v in counter.iteritems() if v > Minfreq}
	return c2, set(c2.keys())

def parseAndSelect(doc):
	'''
	Parses xml file and selects relevant nodes
	:param doc: string, xml doc from which info is gonna be extracted
	:return: list of nodes that represent sentences
	'''

	root=ET.parse(doc)
	sentList=[]
	sentences=root.iter('sentence')
	for s in sentences:

		#verbInfo=s.find(".//lexical")
		#verb=verbInfo.attrib["verb"]
		#sense=verbInfo.attrib["sense"]
		#name='_'.join([verb,sense])
		#if name in ["abrir_18","cerrar_19","crecer_1","dormir_1","escuchar_1","estar_14","explicar_1","gustar_1","gestionar_1", "montar_2","morir_1", "parecer_1", "pensar_2", "perseguir_1","trabajar_1","valer_1","valorar_2","ver_1","viajar_1","volver_1"]:

		sentList.append(s)

	return sentList






def getGeneralInfo(sentences, grupo_info,tipo_info, nivel, clusters):
	'''
	Create a dict that contains the info about co-occurrences

	:param sentences: list of xml nodes which are sentences
	:param grupo_info: str, subcats, constituents or isolated feats
	:param tipo_info: list, contains info to be included
	:param nivel: str, type of unit to be represented (sense, sentence lemma, different types of argument structures)
	:param clusters: unpickled object (dict), clusters created with wordEmbeddings
	:return: dict, keys are units (sense, etc), values are dicts of feats
	'''

	global prep


	generalDic={}

	for s in sentences:

		#prepare for level
		RolesCon = []
		RolesMed = []
		RolesAbs = []

		#sentence info
		infoFraseL=[]

		#sentence level info
		sense = s.find(".//lexical").attrib["sense"]
		vlema = s.find(".//lexical").attrib['verb']
		ide=s.attrib['id']
		#print ide
		perifrastico = s.find(".//lexical[@periphrastic]")

		if 'perif' in tipo_info:
			infoFraseL.append('p_'+perifrastico.attrib['periphrastic'])


		if 'aspectuality' in s.find(".//semantics").attrib:
			aspectualidad = s.find(".//semantics").attrib['aspectuality']
			if 'aspectual' in tipo_info:
				infoFraseL.append(aspectualidad)


		if 'modality' in s.find(".//semantics").attrib:
			modality = s.find(".//semantics").attrib['modality']
			if 'modal' in tipo_info:
				infoFraseL.append(modality)



		if 'polarity' in s.find(".//semantics").attrib:
			polarity = s.find(".//semantics").attrib['polarity']
			if 'polar' in tipo_info:
				infoFraseL.append(polarity)


		aspect1 = s.find(".//argumental").attrib['aspect']
		aspect = 'stative' if aspect1 == 'State' else 'dynamic'

		if 'aspect' in tipo_info:
			infoFraseL.append(aspect)

		if 'construction' in s.find(".//argumental").attrib:
			cons1=[]
			constr = s.find(".//argumental").attrib['construction']
			if constr != None:
				constS = constr.split('-')
				for cc in constS:
					if cc != '' and cc != ' ':
						cc=cc.strip()
						cons1.append(cc)
			c='*'.join(cons1)
			if 'const' in tipo_info:
				infoFraseL.append(c)

		#argument level info
		roles=s.findall('.//phr[@rs]')
		#print 'roles', roles
		for a in roles:
			if a.attrib['arg']=='Argument': #if it is an argument
				ArgFrase=[]

				#gather role equivalences for the diff levels of abstraction
				rol1=a.attrib["rs"]
				if rol1 != '':
					rolSensem =rol1 +'_SS' #specific
					RolesCon.append(rolSensem)
					rolLiricsMed ='-'.join(roles_mapping[rol1].split('-')[:2])+'_SS' #medium
					RolesMed.append(rolLiricsMed)
					rolLiricsAbs =roles_mapping[rol1].split('-')[0]+'_SS'#abstract
					RolesAbs.append(rolLiricsAbs)				
				
					#if the semantic roles are in the info demanded
					if 'RA' in tipo_info:
						ArgFrase.append(rolLiricsAbs)
					if 'RM' in tipo_info:
						ArgFrase.append(rolLiricsMed)
					if 'RC' in tipo_info:
						ArgFrase.append(rolSensem)
						
				#get syntactic function
				syntax = a.attrib["fs"]+'_SS'
				if 'syntax' in tipo_info:
					ArgFrase.append(syntax)

				#get syntactic function	+ selectional preferences
				if 'syntaxSP' in tipo_info:

					clustersA = []
					if a.attrib["fs"] in ['Subject', "Direct obj.", "Indirect obj."]:
						wordS = a.findall('.//word[@core]')
						for w in wordS:

							try: #retrieve cluster ID of word
								clID = clusters[w.text.lower().decode('utf-8')]
								clustersA.append(str(clID))

							except:
								clustersA.append('OOS') #word was not in embeddings

					SPInfo = '*'.join(clustersA)
					#print syntax+'*'+SPInfo.encode('utf-8')
					ArgFrase.append(syntax+'*'+SPInfo.encode('utf-8'))
					
					
				#get syntactic category
				morfo = a.attrib["cat"]+'_SS'
				if 'morfo' in tipo_info:
					ArgFrase.append(morfo)

				# get syntactic category plus prepositional preference
				if 'morfSPrep' in tipo_info:
					prepo = ''
					if morfo[:2] == 'PP':
						subwords = a.findall('.//word')
						prepos = subwords[0].text.lower()#.decode('UTF-8')
						
						if prepos.split('_')[0] in prep:
							prepo = prepos.split('_')[0]
						if prepos == 'al':
							prepo = 'a'
						if prepos == 'del':
							prepo = 'de'


					ArgFrase.append(morfo+'*'+prepo.encode('utf-8'))

				# get syntactic category + selectional preferences
				if 'morfSPref' in tipo_info:

					clustersA = []
					if a.attrib["cat"] in ['NP','InfSC']:
						wordS = a.findall('.//word[@core]')
						for w in wordS:

							try:
								clID = clusters[w.text.lower().decode('utf-8')]
								clustersA.append(str(clID))

							except:
								clustersA.append('OOS')
								#print w.text.lower(), 'OOS'
					SPInfo = '*'.join(clustersA)

					ArgFrase.append(morfo+'*'+SPInfo.encode('utf-8'))	

				# get syntactic category + selectional preferences +prepositional preferences
				if 'morfoSPrepSPref' in tipo_info:
					prepo = ''

					
					if morfo[:2] == 'PP':
						subwords = a.findall('.//word')
						prepos = subwords[0].text.lower()#.decode('UTF-8')
						
						if prepos.split('_')[0] in prep:
							prepo = prepos.split('_')[0]
						if prepos == 'al':
							prepo = 'a'
						if prepos == 'del':
							prepo = 'de'

					clustersA = []
					if a.attrib["cat"] in ['NP','InfSC']:
						wordS = a.findall('.//word[@core]')
						for w in wordS:

							try:
								clID = clusters[w.text.lower().decode('utf-8')]
								clustersA.append(str(clID))

							except:
								clustersA.append('OOS')

					
					SPInfo = '*'.join(clustersA)
					#print SPInfo

					ArgFrase.append(morfo+'*'+prepo.encode('utf-8')+'*'+SPInfo.encode('utf-8'))				

				#retrieve ontological category of words
				wordsS = a.findall('.//word[@core]')
				TCOL=[]
				TCOSplitL=[]
				supersenseL=[]
				sumoL=[]
				lemaL=[]
				for w in wordsS:
					if 'sumo' in w.attrib:	
						sumo=w.attrib['sumo']
						sumoL.append(sumo)	

					if 'supersense' in w.attrib:
						supersense=w.attrib['supersense']
						supersenseL.append(supersense)
						
					if "TCO" in w.attrib:
						tco = w.attrib["TCO"]
						TCOL.append(tco)
						temp=tco.split(' ')
						for e in temp:

							TCOSplitL.append(e)
					
					if 'lemma' in w.attrib:
						lema = w.attrib['lemma']
						lemaL.append(lema)
				

				if 'tco' in tipo_info:
					if TCOL !=[]:
						ArgFrase.append(TCOL)

				if 'tco-split' in tipo_info:
					if TCOSplitL !=[]:
						ArgFrase.append(TCOSplitL)

				if 'sumo' in tipo_info:
					if sumoL !=[]:
						ArgFrase.append(sumoL)
				if 'lema' in tipo_info:
					if lemaL !=[]:
						ArgFrase.append(lemaL)
						
				if 'supersense' in tipo_info:
					if supersenseL !=[]:
						ArgFrase.append(supersenseL)

				infoFraseL.append(ArgFrase) # we add this argument as a list
				#syntactic info is string, semantic is a list

		#print 'argumental info',infoFraseL
		#print vlema+'_'+sense, infoFraseL #[['Indirect obj._SS'], ['Subject_SS', ['Artifact Object Place']]]
		
		###add key to dict deppending on unit selected
		#frase = [info frase [info arg [info onto]]]
		if nivel == 'sentido':
			key = vlema+'_'+sense
				
		if nivel == 'frase':
			key = vlema+'_'+sense+'_'+ide
		
		if nivel == 'EAsenseABS':
			r='-'.join(RolesAbs)
			key = vlema+'_'+sense + '_'+r
		if nivel == 'EAsenseMED':
			r='-'.join(RolesMed)
			key = vlema+'_'+sense + '_'+r

		if nivel == 'EAsenseSE':
			r='-'.join(RolesCon)
			key = vlema+'_'+sense + '_'+r
			
		if nivel == 'lema':
			key = vlema
		
		if key not in generalDic:
			generalDic[key]={}
		
		
		if grupo_info == 'constituyentes' or grupo_info == 'patrones':
			
			listconst=[]#constituyentes de frase
			
			I_oracional = []
			
			for i in infoFraseL: #info oracional 
				if type(i) == str:
					I_oracional.append(i)

			
			for i in infoFraseL:#for argument
				#print 'i--', i

				temp=[]

				if type(i) != str:# it is an argument ['Subject_SS', ['Artifact Object Place']];[['Indirect obj._SS'[lll]], ['Subject_SS'[lll]]]
					simple=[] #syntactic info
					complejo=[] #semantic info at word level
						
					for info in i:
						if type(info) == str: #if it is syntactic
							simple.append(info)
												
						else:#it is semantic info
							
							for lexOnto in info:
								if simple != []:					
									for o in simple:#associate each semantic info with the syntactic info
										t = o+'-'+lexOnto
										complejo.append(t)
								
								#if there is not syntactic info recovered
								if simple == []:
									complejo.append(lexOnto)
									
								

					if complejo != []:
						for c in complejo:
							
							if I_oracional != []:
								for i in I_oracional:
									i2=i+'-'+c
									temp.append(i2)
									#print 'i2', i2
							else:
							
								temp.append(c)
						#print 'tempC',temp

						listconst.append(temp) #adding the argument
							

								
					else:
						#print 'simple'
						for r in simple: #when there is just syntactic info or roles

							
							if I_oracional != []:	
								for i in I_oracional:
									i2=i+'-'+r
									#print 'i2simple',i2
									temp.append(i2)
							else:
							
								temp.append(r)
						#print 'tempS',temp
						
						listconst.append(temp)
			#print 'final', listconst



			if grupo_info == 'constituyentes': #create constituents
				if 'tco-split' in tipo_info: #tco features are added to each constituent
					for cons1 in listconst: #for each sublist (a constituent)
						for cons in cons1:
							if cons in generalDic[key]:
								generalDic[key][cons] +=1
							else:
								generalDic[key][cons] =1					
						
					
				else:
					for cons1 in listconst:
						cons='-'.join(cons1) #join tco features

						if cons in generalDic[key]:
							generalDic[key][cons] +=1
						else:
							generalDic[key][cons] =1
			else:
				if 'tco-split' in tipo_info:
					d=list(itertools.product(*listconst)) #a pattern is created for each feature combination given by the tco feature
					for pat1 in d:
						pat = '+'.join(pat1)

						 
						if pat in generalDic[key]:
							generalDic[key][pat] +=1
						else:
							generalDic[key][pat] =1	
						
					
					
				else:	

					pat1=[]
					for cons1 in listconst:
						cons='-'.join(cons1)
						pat1.append(cons)
					pat='+'.join(pat1)				
					if pat in generalDic[key]:
						generalDic[key][pat] +=1
					else:
						generalDic[key][pat] =1								
							

			
		if grupo_info == 'rasgos':
			for i in infoFraseL:
				if type(i) == str:
					if i in generalDic[key]:
						generalDic[key][i] +=1
					else:
						generalDic[key][i] =1
				else:
					#hay argumentos
					for arg_i in i:
						if type(arg_i) == str:
							if arg_i in generalDic[key]:
								generalDic[key][arg_i] +=1
							else:
								generalDic[key][arg_i] =1
								
						else:
							#estamos en el nivel de la onto
							for onto in arg_i:
								if onto in generalDic[key]:
									generalDic[key][onto] +=1
								else:
									generalDic[key][onto] =1
									

	return generalDic
								
									 


def splitDict(d): ###not used
	n = len(d) // 2          # length of smaller half
	i = iter(d.items())      # alternatively, i = d.iteritems() works in Python 2

	d1 = dict(itertools.islice(i, n))   # grab first n items
	d2 = dict(i)                        # grab the rest
	
	dicty2 = sorted(d1.items(), key=operator.itemgetter(0))
	newd2 = OrderedDict()
	for tup in dicty2:
		newd2[tup[0]]=tup[1]
	
	dictyP2 = sorted(d2.items(), key=operator.itemgetter(0))
	newdP2 = OrderedDict()
	for tup in dictyP2:
		newdP2[tup[0]]=tup[1]
	return newd2, newdP2


def getSet(dicGeneral): ## not used
	s= set()
	for verbo in dicGeneral:
		s=s.union(set(dicGeneral[verbo].keys()))
	return s
	
							
	
	
def to_matrix(setoc,dic,out,param,dataType):
	'''
	Materializes co-occurrence data in csv file
	:param setoc: set with the features that are considered
	:param dic: dict that contains coocurrence info
	:param out: folder for output
	:param param: name for the file
	:param dataType: type of the data (probabilities, frequencies, binary)
	:return: matrix
	'''
	eqs = {'prob':'probabilities','freq':'raw','bin':'bin'}

	if ' ' in setoc:
		setoc.remove(' ')
		
	if '' in setoc:
		setoc.remove('')

	z=sorted(setoc)#sort feats
	z.insert(0,"head")
	z.insert(len(setoc)+1,'none') ##in case a unit has no feats
	#print z
	matrix = []
	matrix.append(z)

	if 'bin' in dataType:

		for verbo in dic:
			vec = [verbo.encode("utf-8")]
			for cons1 in z[1:-1]:
				if cons1 in dic[verbo]:
					vec.append(1)
				else:
					vec.append(0)
			#if the unit had no features associated
			suma = sum(vec[1:])
			if suma != 0:
				vec.append(0)
			else:
				vec.append(1)
			matrix.append(vec)

	elif 'freq' in dataType:
		for verbo in dic:
			vec = [verbo.encode("utf-8")]

			for cons1 in z[1:-1]:
			# for cons1 in z[1:]: ## if sentence level is used

				if cons1 in dic[verbo]:
					vec.append(dic[verbo][cons1])
				else:
					vec.append(0)

			suma = sum(vec[1:])
			if suma != 0:
				vec.append(0)
			else:
				vec.append(1)
			matrix.append(vec)

	else: #probabilities, unit normalization
		for verbo in dic:
			vec1 = []
			for cons1 in z[1:-1]:
				# for cons1 in z[1:]: ## if sentence level is used

				if cons1 in dic[verbo]:
					vec1.append(dic[verbo][cons1])
				else:
					vec1.append(0)

			suma = sum(vec1)
			if suma != 0:
				vec1.append(0)
			else:
				vec1.append(1)

			a = np.array(vec1).reshape(1,-1)
			vec2= preprocessing.normalize(a.reshape(1, -1), norm='l1', axis=1)
			vec = vec2[0].tolist()
			vec.insert(0,verbo.encode("utf-8"))
			matrix.append(vec)

	#### to file
	b = open(out+param+'_{0}.csv'.format(eqs[dataType]), 'w')
	a = csv.writer(b)
	a.writerows(matrix)
	b.close()


	#return freqmatrix, probmatrix, binmatrix
	

#doc='/media/lara/Disc Personal/Google Drive/phd_projects/recursosSensem/sensemFreeling/entero/semTOT.xml' #muestra
#doc = path_uso+'/phd_projects/recursosSensem/sensemFreeling2/sensemMin2.xml'



#doctag='/media/lara/Disc Personal/Google Drive/phd_projects/recursosSensem/sensemFreeling2/miniproba.xml'


#outPAT='/media/lara/Disc Personal/Google Drive/phd_projects/SRL_CV/csvRoles20v/'
#out = path_uso+'/phd_projects/SRL_CV/csvRoles20_2/'
#out = path_uso + '/phd_projects/SRL_CV/csvRoles20_2extra/'
###
#outTraining = path_uso + '/phd_projects/proyectos/data/csvs/conArgs/training/'
#outTestNV = path_uso + '/phd_projects/proyectos/data/csvs/conArgs/testNV/'
#outTestFrasesV = path_uso + '/phd_projects/proyectos/data/csvs/conArgs/testV/'



'''
##ejemplo
dic4tag = getGeneralInfo(allTags, 'patrones',['morfo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['morfo'], 'sentido')
name = 'morfo_pat_paraAnalisis'
print name
to_matrix(setoc,dic,out,name)
'''
'''
dic4tag = getGeneralInfo(allTags, 'patrones',['syntax','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['syntax','lema'], 'sentido')
name = 'syntax_lema_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)
'''

#path_clusters = path_uso + '/phd_projects/proyectos/clustering/featureSet_generators/ClassFromEmbed/ToUse/'

#Dclusters = open(path_uso + '/phd_projects/proyectos/clustering/featureSet_generators/resultsFT7000.txt', 'r')
#clusters = pickle.load(Dclusters)

## posibles argumentos para listaInfo:
## perif aspectuality modality polarity
## aspect construction
## syntax syntaxSP
## morfo morfSPrep morfSPref morfoSPrepSPref
## tco tco-split sumo lema supersense



#def main(listaInfo,formato,ListaUmbralMin,path_clusters):
def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-su','--supra_arg',  nargs='+', choices=['perif','aspectual','modal','polar','const'], help='Sentence level information')
	parser.add_argument('-se', '--semantic_info',  choices=['tco', 'tco-split', 'sumo', 'lema', 'supersense', 'RA', 'RM', 'RC'], help='Semantic info to be included')
	parser.add_argument('-sy','--syntactic_info',  choices=['syntax', 'syntaxSP', 'morfo', 'morfSPrep', 'morfSPref','morfoSPrepSPref'], help='Syntactic info to be included')

	group = parser.add_mutually_exclusive_group()
	group.add_argument('--asp', action='store_true', help='include aspect')
	group.add_argument('--noasp', action='store_true', help='do  not include aspect')

	parser.add_argument('formato', choices=['patrones','constituyentes','rasgos'],help='Formalization of the information (patternts, constituents, features)')
	parser.add_argument('-u','--unit', choices=['sentido','lema','frase','EAsenseABS','EAsenseMED','EAsenseSE'], default='sentido', help='Type of unit: sense, lemma, sentence, argument structure (abstract, medium or concrete roles)')
	parser.add_argument('thresholds',  nargs='+', type=int, help='Minimum frequencies to take into account')
	parser.add_argument('data_type',  choices=['prob', 'freq', 'bin'], help='Type of data of the output (probabilities, frequencies, binary)')

	parser.add_argument('-o', '--output',  default='../GeneratedData/aspect2/',
						help='Output folder for the training/text csv files')
	parser.add_argument('-i', '--input',  default='../RawData/SensemTestTrain/',
						help='Input folder for the sensem files')
	parser.add_argument('-corpus', '--corpus_file',  default='../RawData/sensemMin2.xml', help='Corpus sensem')

	parser.add_argument('-clusters','--path_clusters', default='../Auxdata/ToUse/bueno/', help='Folder that contains the different WE to use. Mandatory for "syntaxSP", "morfSPref", "morfoSPrepSPref"')
	args = parser.parse_args()
	print args

	#set folders & files
	outTraining = args.output + 'training/'
	if not os.path.exists(outTraining):
		os.makedirs(outTraining)

	outTestNV = args.output+'testNV/'
	if not os.path.exists(outTestNV):
		os.makedirs(outTestNV)
	outTestFrasesV = args.output+'testV/'
	if not os.path.exists(outTestFrasesV):
		os.makedirs(outTestFrasesV)
	
	doctag = args.corpus_file
	path_clusters = args.path_clusters
	
	sensesNV = args.input+'testNoVistos5f.xml'
	frasesV = args.input+'testVistos5f.xml'
	training = args.input+'training5f.xml'

	#set parameters
	formato = args.formato
	ListaUmbralMin = args.thresholds
	dataType = args.data_type

	#adding linguistic feats
	listaInfo = []
	if not args.supra_arg == None:
		for el in args.supra_arg:
			listaInfo.append(el)
		formato = 'rasgos'
		print 'For supra argumental info only isolated features (is) are allowed. Changing {} to is'.format(args.formato)

	listaInfo.append(args.semantic_info)
	listaInfo.append(args.syntactic_info)
	listaInfo = filter(None, listaInfo)

	if args.asp:
		listaInfo.insert(0, 'aspect')

	#----start
	corpus = Corpus(doctag)
	corpus.allSentences()
	allTags = corpus.sentence_list #todas las frases
	
	listaClusterSensit = ['syntaxSP', 'morfSPref', 'morfoSPrepSPref']


	ClustersNeeded = next((True for item in listaClusterSensit if item in listaInfo), False)

	# if selectional preferences were included
	if ClustersNeeded:
		for fileCl in os.listdir(path_clusters):
			print fileCl
			
			Dclusters = open(path_clusters +fileCl, 'r')
			clusters = pickle.load(Dclusters)
			

			for tipoDataset in [(outTraining,training),(outTestNV,sensesNV),(outTestFrasesV,frasesV)]:
				outDir = tipoDataset[0]
				fuente = tipoDataset[1]
			
				relevant = parseAndSelect(fuente)
				#----

				dic4tag = getGeneralInfo(allTags, formato,listaInfo, args.unit, clusters)

				
				for umbralMin in ListaUmbralMin:
					reducedDict, setoc = removeLowFrequentFeats(dic4tag,umbralMin)

					dic = getGeneralInfo(relevant, formato,listaInfo, args.unit, clusters)
					nombre = '_'.join(listaInfo)
					clInfo = fileCl.split('.')[0]
					name = '{0}_{1}_{2}_{3}'.format(nombre,clInfo,umbralMin,formato)
					print 'created', name, 'with {0} features in {1}'.format(len(reducedDict), outDir)
					to_matrix(setoc,dic,outDir,name, dataType)
	else:
		print 'not loading WE clusters'
		for tipoDataset in [(outTraining,training),(outTestNV,sensesNV),(outTestFrasesV,frasesV)]:
			outDir = tipoDataset[0]
			fuente = tipoDataset[1]
		
			relevant = parseAndSelect(fuente)
			#----

			dic4tag = getGeneralInfo(allTags, formato,listaInfo, args.unit,None)
			#print dic4tag
			
			for umbralMin in ListaUmbralMin:
				reducedDict, setoc = removeLowFrequentFeats(dic4tag,umbralMin)
				#print setoc
				dic = getGeneralInfo(relevant, formato,listaInfo, args.unit,None)
				nombre = '_'.join(listaInfo)
				clInfo = 'noCl'
				name = '{0}_{1}_{2}_{3}'.format(nombre,clInfo,umbralMin, formato)
				print 'created', name, 'with {0} features in {1}'.format(len(reducedDict),outDir)
				to_matrix(setoc,dic,outDir,name, dataType)
		
		
## posibles argumentos para listaInfo:
## perif aspectuality modality polarity
## aspect construction
## syntax syntaxSP
## morfo morfSPrep morfSPref morfoSPrepSPref
## tco tco-split sumo lema supersense

#main(['aspect','syntaxSP'],'patrones',[2,5,10,50,100],path_clusters)
if __name__ == '__main__':
	main()

'''
extractorDef.py  syntax patrones 0 1 --asp

main(['syntaxSP'],'patrones',[2,3,4,5,10,50,100],path_clusters)
main(['morfSPref'],'patrones',[2,3,4,5,10,50,100],path_clusters)
main(['morfoSPrepSPref'],'patrones',[2,3,4,5,10,50,100],path_clusters)

main(['syntaxSP'],'constituyentes',[2,3,4,5,10,50,100],path_clusters)
main(['morfSPref'],'constituyentes',[2,3,4,5,10,50,100],path_clusters)
main(['morfoSPrepSPref'],'constituyentes',[2,3,4,5,10,50,100],path_clusters)

###tradicionales
main(['syntax','tco'],'constituyentes',[5,10,50,100],path_clusters)
main(['syntax','tco-split'],'constituyentes',[5,10,50,100],path_clusters)
main(['syntax','sumo'],'constituyentes',[5,10,50,100],path_clusters)
main(['syntax','supersense'],'constituyentes',[5,10,50,100],path_clusters)

main(['morfo','tco'],'constituyentes',[5,10,50,100],path_clusters)
main(['morfo','tco-split'],'constituyentes',[5,10,50,100],path_clusters)
main(['morfo','sumo'],'constituyentes',[5,10,50,100],path_clusters)
main(['morfo','supersense'],'constituyentes',[5,10,50,100],path_clusters)

main(['syntax','tco'],'patrones',[5,10,50,100],path_clusters)
main(['syntax','tco-split'],'patrones',[5,10,50,100],path_clusters)
main(['syntax','sumo'],'patrones',[5,10,50,100],path_clusters)
main(['syntax','supersense'],'patrones',[5,10,50,100],path_clusters)

main(['morfo','tco'],'patrones',[5,10,50,100],path_clusters)
main(['morfo','tco-split'],'patrones',[5,10,50,100],path_clusters)
main(['morfo','sumo'],'patrones',[5,10,50,100],path_clusters)
main(['morfo','supersense'],'patrones',[5,10,50,100],path_clusters)

###asp
main(['aspect','syntax','tco'],'constituyentes',[5,10,50,100],path_clusters)
main(['aspect','syntax','tco-split'],'constituyentes',[5,10,50,100],path_clusters)
main(['aspect','syntax','sumo'],'constituyentes',[5,10,50,100],path_clusters)
main(['aspect','syntax','supersense'],'constituyentes',[5,10,50,100],path_clusters)

main(['aspect','morfo','tco'],'constituyentes',[5,10,50,100],path_clusters)
main(['aspect','morfo','tco-split'],'constituyentes',[5,10,50,100],path_clusters)
main(['aspect','morfo','sumo'],'constituyentes',[5,10,50,100],path_clusters)
main(['aspect','morfo','supersense'],'constituyentes',[5,10,50,100],path_clusters)

main(['aspect','morfSPrep','tco'],'constituyentes',[5,10,50,100],path_clusters)
main(['aspect','morfSPrep','tco-split'],'constituyentes',[5,10,50,100],path_clusters)
main(['aspect','morfSPrep','sumo'],'constituyentes',[5,10,50,100],path_clusters)
main(['aspect','morfSPrep','supersense'],'constituyentes',[5,10,50,100],path_clusters)

main(['aspect','syntax','tco'],'patrones',[5,10,50,100],path_clusters)
main(['aspect','syntax','tco-split'],'patrones',[5,10,50,100],path_clusters)
main(['aspect','syntax','sumo'],'patrones',[5,10,50,100],path_clusters)
main(['aspect','syntax','supersense'],'patrones',[5,10,50,100],path_clusters)

main(['aspect','morfo','tco'],'patrones',[5,10,50,100],path_clusters)
main(['aspect','morfo','tco-split'],'patrones',[5,10,50,100],path_clusters)
main(['aspect','morfo','sumo'],'patrones',[5,10,50,100],path_clusters)
main(['aspect','morfo','supersense'],'patrones',[5,10,50,100],path_clusters)

main(['aspect','morfSPrep','tco'],'patrones',[5,10,50,100],path_clusters)
main(['aspect','morfSPrep','tco-split'],'patrones',[5,10,50,100],path_clusters)
main(['aspect','morfSPrep','sumo'],'patrones',[5,10,50,100],path_clusters)
main(['aspect','morfSPrep','supersense'],'patrones',[5,10,50,100],path_clusters)
'''

##########################
'''
#----
allTags = allSentences(doctag)
relevant = parseAndSelect(frasesV) #training, 
#----

dic4tag = getGeneralInfo(allTags, 'constituyentes',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
#print setoc
dic = getGeneralInfo(relevant, 'constituyentes',['RC', 'syntax'], 'sentido')
name = 'RC-syn_cons'
#print name
to_matrix(setoc,dic,outTestFrasesV,name)

#######################
#----
allTags = allSentences(doctag)
relevant = parseAndSelect(sensesNV) #training, 
#----

dic4tag = getGeneralInfo(allTags, 'constituyentes',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
#print setoc
dic = getGeneralInfo(relevant, 'constituyentes',['RC', 'syntax'], 'sentido')
name = 'RC-syn_cons'
#print name
#to_matrix(setoc,dic,outTestNV,name)
'''
####hasta aqui


'''

#con roles, patrones
dic4tag = getGeneralInfo(allTags, 'patrones',['RA'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RA'], 'sentido')
name = 'RA_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RM'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RM'], 'sentido')
name = 'RM_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RC'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RC'], 'sentido')
name = 'RC_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


#con roles, const
dic4tag = getGeneralInfo(allTags, 'constituyentes',['RA'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RA'], 'sentido')
name = 'RA_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['RM'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RM'], 'sentido')
name = 'RM_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['RC'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RC'], 'sentido')
name = 'RC_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)



#con roles, rasgos
dic4tag = getGeneralInfo(allTags, 'rasgos',['RA'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RA'], 'sentido')
name = 'RA_rasgo'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RM'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RM'], 'sentido')
name = 'RM_rasgo'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RC'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RC'], 'sentido')
name = 'RC_rasgo'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


#con roles + syn, patrones
dic4tag = getGeneralInfo(allTags, 'patrones',['RA', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RA', 'syntax'], 'sentido')
name = 'RA-syn_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RM', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RM', 'syntax'], 'sentido')
name = 'RM-syn_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RC', 'syntax'], 'sentido')
name = 'RC-syn_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


#con roles +syn, const
dic4tag = getGeneralInfo(allTags, 'constituyentes',['RA', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
print setoc
dic = getGeneralInfo(relevant, 'constituyentes',['RA', 'syntax'], 'sentido')
name = 'RA-syn_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['RM', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
print setoc
dic = getGeneralInfo(relevant, 'constituyentes',['RM', 'syntax'], 'sentido')
name = 'RM-syn_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)


#dic = getGeneralInfo(relevant, 'rasgos',['RC', 'syntax'], 'sentido')
#name = 'RC-syn_cons'
#print name
#to_matrix(setoc,dic,outTestFrasesV,name)


#con roles +syn, rasgo
dic4tag = getGeneralInfo(allTags, 'rasgos',['RA', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RA', 'syntax'], 'sentido')
name = 'RA-syn_rasgo'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RM', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RM', 'syntax'], 'sentido')
name = 'RM-syn_rasgo'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RC', 'syntax'], 'sentido')
name = 'RC-syn_rasgo'
print name
to_matrix(setoc,dic,outTestFrasesV,name)



#### training

#----
allTags = allSentences(doctag)
relevant = parseAndSelect(training)
#----

#con roles, patrones
dic4tag = getGeneralInfo(allTags, 'patrones',['RA'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RA'], 'sentido')
name = 'RA_pat'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RM'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RM'], 'sentido')
name = 'RM_pat'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RC'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RC'], 'sentido')
name = 'RC_pat'
print name
to_matrix(setoc,dic,outTraining,name)


#con roles, const
dic4tag = getGeneralInfo(allTags, 'constituyentes',['RA'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RA'], 'sentido')
name = 'RA_cons'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['RM'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RM'], 'sentido')
name = 'RM_cons'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['RC'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RC'], 'sentido')
name = 'RC_cons'
print name
to_matrix(setoc,dic,outTraining,name)



#con roles, rasgos
dic4tag = getGeneralInfo(allTags, 'rasgos',['RA'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RA'], 'sentido')
name = 'RA_rasgo'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RM'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RM'], 'sentido')
name = 'RM_rasgo'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RC'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RC'], 'sentido')
name = 'RC_rasgo'
print name
to_matrix(setoc,dic,outTraining,name)


#con roles + syn, patrones
dic4tag = getGeneralInfo(allTags, 'patrones',['RA', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RA', 'syntax'], 'sentido')
name = 'RA-syn_pat'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RM', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RM', 'syntax'], 'sentido')
name = 'RM-syn_pat'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RC', 'syntax'], 'sentido')
name = 'RC-syn_pat'
print name
to_matrix(setoc,dic,outTraining,name)


#con roles +syn, const
dic4tag = getGeneralInfo(allTags, 'constituyentes',['RA', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RA', 'syntax'], 'sentido')
name = 'RA-syn_cons'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['RM', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
print setoc
dic = getGeneralInfo(relevant, 'constituyentes',['RM', 'syntax'], 'sentido')
name = 'RM-syn_cons'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
print setoc
dic = getGeneralInfo(relevant, 'rasgos',['RC', 'syntax'], 'sentido')
name = 'RC-syn_cons'
print name
to_matrix(setoc,dic,outTraining,name)


#con roles +syn, rasgo
dic4tag = getGeneralInfo(allTags, 'rasgos',['RA', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RA', 'syntax'], 'sentido')
name = 'RA-syn_rasgo'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RM', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RM', 'syntax'], 'sentido')
name = 'RM-syn_rasgo'
print name
to_matrix(setoc,dic,outTraining,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RC', 'syntax'], 'sentido')
name = 'RC-syn_rasgo'
print name
to_matrix(setoc,dic,outTraining,name)

####no vistos



#----
allTags = allSentences(doctag)
relevant = parseAndSelect(sensesNV)
#----

#con roles, patrones
dic4tag = getGeneralInfo(allTags, 'patrones',['RA'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RA'], 'sentido')
name = 'RA_pat'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RM'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RM'], 'sentido')
name = 'RM_pat'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RC'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RC'], 'sentido')
name = 'RC_pat'
print name
to_matrix(setoc,dic,outTestNV,name)


#con roles, const
dic4tag = getGeneralInfo(allTags, 'constituyentes',['RA'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RA'], 'sentido')
name = 'RA_cons'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['RM'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RM'], 'sentido')
name = 'RM_cons'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['RC'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RC'], 'sentido')
name = 'RC_cons'
print name
to_matrix(setoc,dic,outTestNV,name)



#con roles, rasgos
dic4tag = getGeneralInfo(allTags, 'rasgos',['RA'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RA'], 'sentido')
name = 'RA_rasgo'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RM'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RM'], 'sentido')
name = 'RM_rasgo'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RC'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RC'], 'sentido')
name = 'RC_rasgo'
print name
to_matrix(setoc,dic,outTestNV,name)


#con roles + syn, patrones
dic4tag = getGeneralInfo(allTags, 'patrones',['RA', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RA', 'syntax'], 'sentido')
name = 'RA-syn_pat'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RM', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RM', 'syntax'], 'sentido')
name = 'RM-syn_pat'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['RC', 'syntax'], 'sentido')
name = 'RC-syn_pat'
print name
to_matrix(setoc,dic,outTestNV,name)


#con roles +syn, const
dic4tag = getGeneralInfo(allTags, 'constituyentes',['RA', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RA', 'syntax'], 'sentido')
name = 'RA-syn_cons'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['RM', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['RM', 'syntax'], 'sentido')
name = 'RM-syn_cons'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RC', 'syntax'], 'sentido')
name = 'RC-syn_cons'
print name
to_matrix(setoc,dic,outTestNV,name)


#con roles +syn, rasgo
dic4tag = getGeneralInfo(allTags, 'rasgos',['RA', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RA', 'syntax'], 'sentido')
name = 'RA-syn_rasgo'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RM', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RM', 'syntax'], 'sentido')
name = 'RM-syn_rasgo'
print name
to_matrix(setoc,dic,outTestNV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['RC', 'syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['RC', 'syntax'], 'sentido')
name = 'RC-syn_rasgo'
print name
to_matrix(setoc,dic,outTestNV,name)
'''

#####

# patrones sin roles
'''
dic4tag = getGeneralInfo(allTags, 'patrones',['morfo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['morfo'], 'sentido')
name = 'morfo_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','morfo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','morfo'], 'sentido')
name = 'morfo_asp_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','syntax'], 'sentido')
name = 'syntax_asp_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['syntax'], 'sentido')
name = 'syntax_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['morfo','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['morfo','sumo'], 'sentido')
name = 'morfo_sumo_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['syntax','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['syntax','sumo'], 'sentido')
name = 'syntax_sumo_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['morfo','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['morfo','supersense'], 'sentido')
name = 'morfo_supersenses_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['syntax','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['syntax','supersense'], 'sentido')
name = 'syntax_supersenses_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['morfo','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['morfo','tco'], 'sentido')
name = 'morfo_tco_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['morfo','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['morfo','tco-split'], 'sentido')
name = 'morfo_tco-split_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['syntax','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['syntax','tco'], 'sentido')
name = 'syntax+tco_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['syntax','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['syntax','tco-split'], 'sentido')
name = 'syntax_tco-split_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['morfo','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['morfo','lema'], 'sentido')
name = 'morfo_lema_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['syntax','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['syntax','lema'], 'sentido')
name = 'syntax_lema_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','morfo','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','morfo','sumo'], 'sentido')
name = 'asp_morfo_sumo_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','syntax','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','syntax','sumo'], 'sentido')
name = 'asp_syntax_sumo_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','morfo','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','morfo','supersense'], 'sentido')
name = 'asp_morfo_supersenses_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','syntax','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','syntax','supersense'], 'sentido')
name = 'asp_syntax_supersenses_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','morfo','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','morfo','tco'], 'sentido')
name = 'asp_morfo_tco_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','syntax','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','syntax','tco'], 'sentido')
name = 'asp_syntax_tco_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','morfo','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','morfo','tco-split'], 'sentido')
name = 'asp_morfo_tco-split_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','syntax','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','syntax','tco-split'], 'sentido')
name = 'asp_syntax_tco-split_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','morfo','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','morfo','lema'], 'sentido')
name = 'asp_morfo_lema_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'patrones',['aspect','syntax','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'patrones',['aspect','syntax','lema'], 'sentido')
name = 'asp_syntax_lema_pat'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


# cons sin roles



dic4tag = getGeneralInfo(allTags, 'constituyentes',['morfo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['morfo'], 'sentido')
name = 'morfo_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','morfo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','morfo'], 'sentido')
name = 'morfo_asp_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','syntax'], 'sentido')
name = 'syntax_asp_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['syntax'], 'sentido')
name = 'syntax_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['morfo','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['morfo','sumo'], 'sentido')
name = 'morfo_sumo_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['syntax','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['syntax','sumo'], 'sentido')
name = 'syntax_sumo_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['morfo','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['morfo','supersense'], 'sentido')
name = 'morfo_supersenses_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['syntax','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['syntax','supersense'], 'sentido')
name = 'syntax_supersenses_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['morfo','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['morfo','tco'], 'sentido')
name = 'morfo_tco_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['morfo','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['morfo','tco-split'], 'sentido')
name = 'morfo_tco-split_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['syntax','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['syntax','tco'], 'sentido')
name = 'syntax+tco_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['syntax','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['syntax','tco-split'], 'sentido')
name = 'syntax_tco-split_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['morfo','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['morfo','lema'], 'sentido')
name = 'morfo_lema_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['syntax','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['syntax','lema'], 'sentido')
name = 'syntax_lema_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','morfo','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','morfo','sumo'], 'sentido')
name = 'asp_morfo_sumo_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','syntax','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','syntax','sumo'], 'sentido')
name = 'asp_syntax_sumo_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','morfo','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','morfo','supersense'], 'sentido')
name = 'asp_morfo_supersenses_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','syntax','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','syntax','supersense'], 'sentido')
name = 'asp_syntax_supersenses_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','morfo','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','morfo','tco'], 'sentido')
name = 'asp_morfo_tco_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','syntax','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','syntax','tco'], 'sentido')
name = 'asp_syntax_tco_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','morfo','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','morfo','tco-split'], 'sentido')
name = 'asp_morfo_tco-split_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','syntax','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','syntax','tco-split'], 'sentido')
name = 'asp_syntax_tco-split_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','morfo','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','morfo','lema'], 'sentido')
name = 'asp_morfo_lema_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'constituyentes',['aspect','syntax','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'constituyentes',['aspect','syntax','lema'], 'sentido')
name = 'asp_syntax_lema_cons'
print name
to_matrix(setoc,dic,outTestFrasesV,name)



# rasgos sin roles



dic4tag = getGeneralInfo(allTags, 'rasgos',['morfo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['morfo'], 'sentido')
name = 'morfo_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','morfo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','morfo'], 'sentido')
name = 'morfo_asp_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','syntax'], 'sentido')
name = 'syntax_asp_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['syntax'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['syntax'], 'sentido')
name = 'syntax_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['morfo','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['morfo','sumo'], 'sentido')
name = 'morfo_sumo_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['syntax','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['syntax','sumo'], 'sentido')
name = 'syntax_sumo_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['morfo','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['morfo','supersense'], 'sentido')
name = 'morfo_supersenses_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['syntax','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['syntax','supersense'], 'sentido')
name = 'syntax_supersenses_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['morfo','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['morfo','tco'], 'sentido')
name = 'morfo_tco_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['morfo','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['morfo','tco-split'], 'sentido')
name = 'morfo_tco-split_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['syntax','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['syntax','tco'], 'sentido')
name = 'syntax+tco_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['syntax','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['syntax','tco-split'], 'sentido')
name = 'syntax_tco-split_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['morfo','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['morfo','lema'], 'sentido')
name = 'morfo_lema_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['syntax','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['syntax','lema'], 'sentido')
name = 'syntax_lema_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','morfo','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','morfo','sumo'], 'sentido')
name = 'asp_morfo_sumo_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','syntax','sumo'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','syntax','sumo'], 'sentido')
name = 'asp_syntax_sumo_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','morfo','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','morfo','supersense'], 'sentido')
name = 'asp_morfo_supersenses_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','syntax','supersense'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','syntax','supersense'], 'sentido')
name = 'asp_syntax_supersenses_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','morfo','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','morfo','tco'], 'sentido')
name = 'asp_morfo_tco_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','syntax','tco'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','syntax','tco'], 'sentido')
name = 'asp_syntax_tco_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','morfo','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','morfo','tco-split'], 'sentido')
name = 'asp_morfo_tco-split_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','syntax','tco-split'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','syntax','tco-split'], 'sentido')
name = 'asp_syntax_tco-split_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)

dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','morfo','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','morfo','lema'], 'sentido')
name = 'asp_morfo_lema_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)


dic4tag = getGeneralInfo(allTags, 'rasgos',['aspect','syntax','lema'], 'sentido')
setoc = getSet(dic4tag)
dic = getGeneralInfo(relevant, 'rasgos',['aspect','syntax','lema'], 'sentido')
name = 'asp_syntax_lema_rasgos'
print name
to_matrix(setoc,dic,outTestFrasesV,name)
'''
