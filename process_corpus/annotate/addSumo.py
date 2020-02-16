#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lxml import etree as ET
import argparse
import os



# annotate SenSem corpus with sumo ontology, supersenses y TCO


def SupersensestoDict(lexicograferFile):
	dic = {}
	with open(lexicograferFile,'rb') as file:
		for line in file:
			dic[line.split()[0]] = line.split()[1]
	return dic

def TCOtoDict(iliToTCO,synToILI):
	synset_ili = {}
	with open(synToILI,'r') as synfile:
		for line in synfile:
			synset = line.split('\t')[2].strip('"')
			ili = line.split('\t')[0].strip('"')
			if not synset in synset_ili:
				synset_ili[synset]=[ili]
			else:
				synset_ili[synset].append(ili)

	ili_onto = {}
	with open(iliToTCO, 'r') as tcofile:
		for line in tcofile:
			ili = line.split(',')[1].strip('"')
			onto = line.split(',')[0].strip('"')
			if not ili in ili_onto:
				ili_onto[ili] = [onto]
			else:
				ili_onto[ili].append(onto)

	synset_onto = {}
	for syn in synset_ili:
		for ili1 in synset_ili[syn]:
			for ili2 in ili_onto:
				if ili1 == ili2:
					synset_onto[syn]=ili_onto[ili2]
	return synset_onto

def toData(path, digits, supersensesL):
	data = {}
	with open(path, 'r') as file:
		for line in file:
			if line[0] in digits:
				newline=line.split()

				synsetS=newline[0]
				supersense=newline[1]
				sumo=newline[-1]
				if "&%" in sumo:
					sumo=newline[-1].strip("&%")
					if sumo != '':
						if sumo[-1]=='=':
							val='syn_eq'
						if sumo[-1]=='+':
							val="syn_sub"
						if sumo[-1]=='@':
							val="syn_instance"
						sumo=sumo[:-1]
						S=supersensesL[supersense]
						if sumo != '=>' and sumo != '' and sumo != ' ':
							data[synsetS]=([sumo,val,S])
	return data


def tag(sensem_object, pathOutput, TCOdic, SUMO):

	nouns, verbs, advs, adjs = SUMO
	
	words=sensem_object.iter('word')
	for word in words:
		if "synsetFreeling" in word.attrib:
			if "pos" in word.attrib:
				w=word.attrib["synsetFreeling"].split("-")
				if word.attrib["synsetFreeling"] in TCOdic:
					word.set("TCO",TCOdic[word.attrib["synsetFreeling"]][:-1])

					
				if w[1] =='n':
					if w[0] in nouns:
						ontol=nouns[w[0]][0]
						supersens=nouns[w[0]][2]
						word.set("sumo", ontol)
						word.set("supersense",supersens)
						
				if w[1] =='v':
					if w[0] in verbs:
						ontol=verbs[w[0]][0]
						word.set("sumo", ontol)
						supersens=verbs[w[0]][2]
						word.set("supersense",supersens)

				if w[1] =='r':
					if w[0] in advs:
						ontol=advs[w[0]][0]
						word.set("sumo", ontol)
						supersens=advs[w[0]][2]
						word.set("supersense",supersens)
						
				if w[1] =='a':
					if w[0] in adjs:
						ontol=adjs[w[0]][0]
						word.set("sumo", ontol)
						supersens=adjs[w[0]][2]
						word.set("supersense",supersens)
				

	sensem_object.write(os.path.join(pathOutput, 'AnnotatedSensem.xml'), pretty_print=True, encoding="UTF-8")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-su', '--mappingSUMO', help='Folder containing mappings with SUMO (downloadable from https://github.com/ontologyportal/sumo/tree/master/WordNetMappings)')
	parser.add_argument('-le', '--lexifiles', help='Fomder containing Wordnet 3.0')
	parser.add_argument('-mcr', '--mcrfiles', help='MCR folder, downloadable from https://adimen.si.ehu.es/web/MCR')
	parser.add_argument('-i', '--input', help='Sensem corpus')
	parser.add_argument('-o', '--output', help='Outpud folder for annotated corpus')
	args = parser.parse_args()

	#Supersenses
	supersenses = os.path.join(args.lexifiles, "dict", "lexnames")
	supersensesL = SupersensestoDict(supersenses)

	#TCO
	synsetToILI = os.path.join(args.mcrfiles, 'spaWN', 'wei_spa-30_to_ili.csv')
	ilitoTCO = os.path.join(args.mcrfiles,'TopOntology','wei_ili_to_to.tsv')
	TCOdict = TCOtoDict(ilitoTCO,synsetToILI)

	#Sumo
	digits=['0','1','2','3','4','5','6','7','8','9']
	sumo_noun = os.path.join(args.mappingSUMO, "WordNetMappings30-noun.txt")
	sumo_verb = os.path.join(args.mappingSUMO, "WordNetMappings30-verb.txt")
	sumo_adv = os.path.join(args.mappingSUMO, "WordNetMappings30-adv.txt")
	sumo_adj = os.path.join(args.mappingSUMO, "WordNetMappings30-adj.txt")
	noun_vals = toData(sumo_noun, digits, supersensesL)
	adj_vals = toData(sumo_adj, digits, supersensesL)
	adv_vals = toData(sumo_adv, digits, supersensesL)
	verb_vals = toData(sumo_verb, digits, supersensesL)
	SumoInfo = [noun_vals, adj_vals, adv_vals, verb_vals]
	sensem_object = ET.parse(args.input)
	tag(sensem_object,args.output, TCOdict, SumoInfo)

if __name__ == '__main__':
	main()


