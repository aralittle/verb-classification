#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
anotate sensem with synsets
'''

import argparse
from lxml import etree as ET
from subprocess import *
import subprocess



def lemmatize(sent, freelingRoot):
	# config file must be configured to perform word sense desambiguation with mfs (most frequent sense)
	args = ['analyze', '-f', freelingRoot+'/data/config/es.cfg']
	proc = Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
	proc.stdin.write('{0}'.format(sent))
	data = proc.communicate()[0]
	proc.wait()
	return data


def processAnnotation(data):
	infoList = []
	n = data.strip().split("\n")
	for pal in n:
		pal = pal.split()
		if pal != []:
			forma=pal[0]
			lemma=pal[1]
			pos=pal[2]
			syn=pal[4].split(":")[0]
			infoList.append([forma,lemma, pos, syn])
	return infoList


def annotateSenSemSentences(sensem, output, freelingRoot):

	allsent = []
	sentences = sensem.iter("sentence")
	for sentence in sentences:
		sentenceid = sentence.attrib["id"]
		allsent.append(sentenceid)
		words = sentence.iter('word')
		sent=[]
		for w in words:
			if w.text is not None:
				sent.append(w.text)
		sentencetext = " ".join(sent)
		d = lemmatize(sentencetext, freelingRoot)
		data = processAnnotation(d)

		for w in words:
			for t in data[:]:
				if w.text==t[0]:
					w.set("lemma",unicode(t[1]))
					w.set("pos",unicode(t[2]))
					w.set("synsetFreeling",unicode(t[3]))
					
					data.remove(t)
					break

	sensem.write(output, pretty_print=True, encoding="UTF-8")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--freelingDir', help='Freeling directory')
	parser.add_argument('-u', '--ukbDir', help='UKB directory')
	parser.add_argument('-i', '--input', help='Sensem corpus')
	parser.add_argument('-o', '--output', help='Output folder for annotated corpus')
	args = parser.parse_args()

	sensem = ET.parse(args.input)
	annotateSenSemSentences(sensem, args.output, args.freelingDir)
