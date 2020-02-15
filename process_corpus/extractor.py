# -*- coding: utf-8 -*-
import pandas
import argparse
import os
import pickle
from lxml import etree as ET
from collections import Counter


role_mapping = {'Ag/caus': 'Actor-Actor-Actor', 'Goal_ag': 'Actor-Agent-Agent', 'Ag_exp': 'Actor-Agent-Agent',
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

prep = ['a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en' , 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'según', 'sin', 'so', 'sobre', 'tras', 'versus ', 'vía', 'vía']


class Argument:

    def __init__(self, argument, clusters):
        self.argument = argument
        self.clusters = clusters
        self.abstractRole = ''
        self.mediumRole = ''
        self.sensemRole = ''
        self.synFunct = ''
        self.synFunctSelectPref = ''
        self.synCat = ''
        self.synCatPrep = ''
        self.synCatCluster = ''
        self.synCatPrepCluster = ''
        self.ontoTCO = ''
        self.ontoTCOSplit = []
        self.ontoSupersense = ''
        self.ontoSumo = ''
        self.lemma = ''

    def get_roles(self):
        try:
            self.abstractRole = role_mapping[self.argument.attrib['rs']].split('-')[0]+'_SS'
            self.mediumRole = '-'.join(role_mapping[self.argument.attrib['rs']].split('-')[:2])+'_SS'
            self.sensemRole = self.argument.attrib['rs']+'_SS'
        except KeyError:
            print('argument does not have role tag')

    def get_syntactic_function_info(self):
        try:
            self.synFunct = self.argument.attrib["fs"]+'_SS'

            clustersArgument = []
            if self.argument.attrib["fs"] in ['Subject', "Direct obj.", "Indirect obj."]:
                words = self.argument.findall('.//word[@core]')
                for w in words:
                    try:  # retrieve cluster ID of word
                        clusterID = self.clusters[w.text.lower()]#.encode('utf-8')]
                        clustersArgument.append(str(clusterID))
                        #print(str(clusterID))
                    except:
                        clustersArgument.append('OOS')  # word was not in embeddings
            argumentInfo = '*'.join(clustersArgument)
            self.synFunctSelectPref = self.argument.attrib["fs"]+'*'+argumentInfo if clustersArgument != [] else self.argument.attrib["fs"]

        except KeyError:
            print('argument does not have syntactic function tag')

    def get_syntactic_category_info(self, prep):
        try:
            self.synCat = self.argument.attrib["cat"]+'_SS'

            prepoInfo = ''
            if self.synCat[:2] == 'PP':
                subwords = self.argument.findall('.//word')
                preposition = subwords[0].text.lower()  # .decode('UTF-8')
                if preposition.split('_')[0] in prep:
                    prepoInfo = preposition.split('_')[0]
                if preposition in [u'al', 'al']:
                    prepoInfo = u'a'
                if preposition in [u'del', 'del']:
                    prepoInfo = u'de'
            self.synCatPrep = self.argument.attrib["cat"] + '*' + prepoInfo if prepoInfo != '' else self.argument.attrib["cat"]

            clustersArgument = []
            if self.argument.attrib["cat"] in ['NP', 'InfSC']:
                words = self.argument.findall('.//word[@core]')
                for w in words:
                    try:  # retrieve cluster ID of word
                        clusterID = self.clusters[w.text.lower()] #
                        clustersArgument.append(str(clusterID))
                    except:
                        clustersArgument.append('OOS')  # word was not in embeddings
            argumentInfo = '*'.join(clustersArgument)
            self.synCatCluster = self.argument.attrib["cat"] + '*' + argumentInfo if clustersArgument != [] else self.argument.attrib["cat"]

            self.synCatPrepCluster = self.synCatPrep + '*' + argumentInfo if clustersArgument != [] else self.synCatPrep

        except KeyError:
            print('argument does not have syntactic category tag')

    def ontological_info(self):
        # retrieve ontological category of words
        wordsS = self.argument.findall('.//word[@core]')
        TCOList = []
        TCOSplitList = []
        supersenseList = []
        sumoList = []
        lemaList = []
        for w in wordsS:
            if 'sumo' in w.attrib:
                sumo = w.attrib['sumo']
                sumoList.append(sumo)

            if 'supersense' in w.attrib:
                supersense = w.attrib['supersense']
                supersenseList.append(supersense)

            if "TCO" in w.attrib:
                tco = w.attrib["TCO"]
                TCOList.append(tco)

                temp = tco.split(' ')
                for e in temp:
                    TCOSplitList.append(e)

            if 'lemma' in w.attrib:
                lema = w.attrib['lemma']
                lemaList.append(lema)

        self.ontoTCO = '+'.join(TCOList)
        self.ontoTCOSplit = TCOSplitList
        self.ontoSupersense = '+'.join(supersenseList)
        self.ontoSumo = '+'.join(sumoList)
        self.lemma = '+'.join(lemaList)


class Sentence:
    def __init__(self, sentence, clusters):
        self.sentence = sentence
        self.clusters = clusters
        self.sense = ''
        self.lemma = ''
        self.ide = ''
        self.periphrastic = ''
        self.aspectuality = ''
        self.modality = ''
        self.polarity = ''
        self.aspect = ''
        self.construction = ''
        self.arguments = []
        self.key = ''

    def get_sense_lemma_ID(self):
        self.sense = self.sentence.find(".//lexical").attrib["sense"]
        self.lemma = self.sentence.find(".//lexical").attrib['verb']
        self.ide = self.sentence.attrib['id']

    def get_sentence_info(self):
        self.periphrastic = 'p_' + self.sentence.find(".//lexical[@periphrastic]").attrib['periphrastic']
        self.aspectuality = self.sentence.find(".//semantics").attrib['aspectuality']
        self.modality = self.sentence.find(".//semantics").attrib['modality']
        self.polarity = self.sentence.find(".//semantics").attrib['polarity']
        self.aspect = 'stative' if self.sentence.find(".//argumental").attrib['aspect'] == 'State' else 'dynamic'
        self.construction = '*'.join([c.strip() for c in self.sentence.find(".//argumental").attrib['construction'].split('-') if c not in ['', ' ']])
        self.arguments = [arg for arg in self.sentence.findall('.//phr[@rs]') if arg.attrib['arg'] == 'Argument']

    def get_key_for_info_type(self, info_type):
        if self.lemma == '':
            self.get_sense_lemma_ID()
            self.key = self.lemma
        if not self.arguments:
            self.get_sentence_info()

        if info_type == 'sense':
            self.key = self.lemma + '_' + self.sense

        if info_type == 'frase':
            self.key = self.lemma + '_' + self.sense + '_' + self.ide

        RolesAbs = []
        RolesMed = []
        RolesSpec = []
        for arg in self.arguments:
            argument = Argument(arg, self.clusters)
            argument.get_roles()
            RolesAbs.append(argument.abstractRole)
            RolesMed.append(argument.mediumRole)
            RolesSpec.append(argument.sensemRole)

        if info_type == 'EAsenseABS':
            r = '-'.join(RolesAbs)
            self.key = self.lemma + '_' + self.sense + '_' + r

        if info_type == 'EAsenseMED':
            r = '-'.join(RolesMed)
            self.key = self.lemma + '_' + self.sense + '_' + r

        if info_type == 'EAsenseSE':
            r = '-'.join(RolesSpec)
            self.key = self.lemma + '_' + self.sense + '_' + r

        if info_type == 'lema':
            self.key = self.lemma


class Corpus:
    def __init__(self, doc, train_data_doc, test_data_doc, info_type):

        self.doc = doc
        self.train_data_doc = train_data_doc
        self.test_data_doc = test_data_doc
        self.info_type = info_type

        self.train_data = []
        self.test_data = []
        self.sentences = []

        self.syntactic_info = ''
        self.semantic_info = ''
        self.aspect = False
        self.periph = False
        self.aspectuality = False
        self.modality = False
        self.polarity = False
        self.construction = False

        self.concurrences_format = []
        self.data_type = []
        self.num = 1
        self.clusters = ''

        self.dicSentenceFeats = {}
        self.dicFeats = {}
        self.dicCons = {}
        self.dicPattern = {}

    def get_train_test_data(self):
        root = ET.parse(self.train_data_doc)
        sentences = root.iter('sentence')
        for s in sentences:
            sentence = Sentence(s, self.clusters)
            sentence.get_key_for_info_type(self.info_type)
            self.train_data.append(sentence.key)

        root = ET.parse(self.test_data_doc)
        sentences = root.iter('sentence')
        for s in sentences:
            sentence = Sentence(s, self.clusters)
            sentence.get_key_for_info_type(self.info_type)
            self.test_data.append(sentence.key)

    def get_sentences(self):
        root = ET.parse(self.doc)
        sentences = root.iter('sentence')
        for s in sentences:
            sentence = Sentence(s, self.clusters)
            sentence.get_key_for_info_type(self.info_type)
            self.sentences.append(sentence)

    def get_sentence_features(self):
        for sentence in self.sentences:
            if sentence.key not in self.dicSentenceFeats:
                self.dicSentenceFeats[sentence.key] = Counter()

            if self.aspect:
                self.dicSentenceFeats[sentence.key].update([sentence.aspect])

            if self.periph:
                self.dicSentenceFeats[sentence.key].update([sentence.periphrastic])

            if self.aspectuality:
                self.dicSentenceFeats[sentence.key].update([sentence.aspectuality])

            if self.modality:
                self.dicSentenceFeats[sentence.key].update([sentence.modality])

            if self.polarity:
                self.dicSentenceFeats[sentence.key].update([sentence.polarity])

            if self.construction:
                self.dicSentenceFeats[sentence.key].update([sentence.construction])

    def get_argument_features(self):
        for sentence in self.sentences:
            if sentence.key not in self.dicFeats:
                self.dicFeats[sentence.key] = Counter()
                self.dicCons[sentence.key] = Counter()
                self.dicPattern[sentence.key] = Counter()

            sentenceInfo = []

            for arg in sentence.arguments:
                argument = Argument(arg, self.clusters)
                argument.get_roles()

                argInfo = []

                if self.semantic_info:
                    argument.ontological_info()

                    if self.semantic_info == 'TCO':
                        self.dicFeats[sentence.key].update([argument.ontoTCO])
                        argInfo.append(argument.ontoTCO)

                    if self.semantic_info == 'TCOsplit':
                        for feat in argument.ontoTCOSplit:
                            self.dicFeats[sentence.key].update([feat])
                            argInfo.append(feat)

                    if self.semantic_info == 'supersense':
                        self.dicFeats[sentence.key].update([argument.ontoSupersense])
                        argInfo.append(argument.ontoSupersense)

                    if self.semantic_info == 'sumo':
                        self.dicFeats[sentence.key].update([argument.ontoSumo])
                        argInfo.append(argument.ontoSumo)

                    if self.semantic_info == 'lemma':
                        self.dicFeats[sentence.key].update([argument.lemma])
                        argInfo.append(argument.lemma)

                    if self.semantic_info == 'sensem':
                        self.dicFeats[sentence.key].update([argument.sensemRole])
                        argInfo.append(argument.sensemRole)

                    if self.semantic_info == 'medium':
                        self.dicFeats[sentence.key].update([argument.mediumRole])
                        argInfo.append(argument.mediumRole)

                    if self.semantic_info == 'abstract':
                        self.dicFeats[sentence.key].update([argument.abstractRole])
                        argInfo.append(argument.abstractRole)

                if self.syntactic_info:
                    argument.get_syntactic_function_info()

                    if 'synFunct' in self.syntactic_info:
                        if self.syntactic_info == 'synFunct':
                            self.dicFeats[sentence.key].update([argument.synFunct])
                            argInfo.append(argument.synFunct)

                        if self.syntactic_info == 'synFunctSelectPref':
                            self.dicFeats[sentence.key].update([argument.synFunctSelectPref])
                            argInfo.append(argument.synFunctSelectPref)

                    if 'synCat' in self.syntactic_info:
                        argument.get_syntactic_category_info(prep)
                        if self.syntactic_info == 'synCat':
                            self.dicFeats[sentence.key].update([argument.synCat])
                            argInfo.append(argument.synCat)

                        if self.syntactic_info == 'synCatPrep':
                            self.dicFeats[sentence.key].update([argument.synCatPrep])
                            argInfo.append(argument.synCatPrep)

                        if self.syntactic_info == 'synCatCluster':
                            self.dicFeats[sentence.key].update([argument.synCatCluster])
                            argInfo.append(argument.synCatCluster)

                        if self.syntactic_info == 'synCatPrepCluster':
                            self.dicFeats[sentence.key].update([argument.synCatPrepCluster])
                            argInfo.append(argument.synCatPrepCluster)

                cons = '+'.join(argInfo)
                self.dicCons[sentence.key].update([cons])
                sentenceInfo.append(cons)
            pat = '+'.join(sentenceInfo)
            self.dicPattern[sentence.key].update([pat])


    def convert_to_matrix(self, output_path):
        for type in self.data_type:
            data_dic = {}
            if type == 'feats':
                data_dic = self.dicFeats
            if type == 'cons':
                data_dic = self.dicCons
            if type == 'pats':
                data_dic = self.dicPattern
            for key in self.dicSentenceFeats:
                if key in data_dic:
                    for val in self.dicSentenceFeats[key]:
                        data_dic[key].update([val])
                else:
                    data_dic[key] = self.dicSentenceFeats[key]

            df = pandas.DataFrame(data_dic).transpose()
            if ' ' in df.columns:
                df.drop(' ', axis=1, inplace=True)
            if '' in df.columns:
                df.drop('', axis=1, inplace=True)

            df = df.fillna(0)

            df = df[df.columns[df.sum(axis=0) >= self.num]] # remove feats with freq lower than num
            test_elements = [i for i in df.index if i in self.test_data]
            train_elements = [i for i in df.index if i in self.train_data]
            testDataFrame = df.loc[test_elements]
            trainDataFrame = df.loc[train_elements]

            # add column for verbs that do not have any feature
            underrepresentedTest = testDataFrame.sum(axis=1) < 1
            underrepresentedTest = underrepresentedTest.astype(int)
            testDataFrame.insert(len(df.columns), 'none', underrepresentedTest.values)

            underrepresentedTrain = df.sum(axis=1) < 1
            underrepresentedTrain = underrepresentedTrain.astype(int)
            trainDataFrame.insert(len(df.columns), 'none', underrepresentedTrain.values)

            if 'bin' in self.concurrences_format:
                trainDataFrame = trainDataFrame.round(1)
                testDataFrame = testDataFrame.round(1)
                trainDataFrame.to_csv(os.path.join(output_path, 'training/', 'train.bin.csv'))
                testDataFrame.to_csv(os.path.join(output_path, 'testNV', 'test.bin.csv'))

            if 'prob' in self.concurrences_format:
                trainDataFrame = trainDataFrame.div(trainDataFrame.sum(axis=1), axis=0)
                testDataFrame = testDataFrame.div(testDataFrame.sum(axis=1), axis=0)
                trainDataFrame.to_csv(os.path.join(output_path, 'training/', 'train.prob.csv'))
                testDataFrame.to_csv(os.path.join(output_path, 'testNV', 'test.prob.csv'))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-su', '--supra_arg', nargs='+', choices=['aspect','perif', 'aspectual', 'modal', 'polar', 'const'],
                        help='Sentence level information')

    parser.add_argument('-se', '--semantic_info',
                        choices=['TCO', 'TCOsplit', 'sumo', 'lemma', 'supersense', 'abstract', 'medium', 'sensem'],
                        help='Semantic info to be included')

    parser.add_argument('-sy', '--syntactic_info',
                        choices=['synFunct', 'synFunctSelectPref', 'synCat', 'synCatPrep', 'synCatCluster', 'synCatPrepCluster'],
                        help='Syntactic info to be included')

    parser.add_argument('concurrence_type',  nargs='+', choices=['prob', 'bin'],
                        help='Type of data of the output (prob, bin)')


    parser.add_argument('-u', '--unit', choices=['sense', 'lemma', 'frase', 'EAsenseABS', 'EAsenseMED', 'EAsenseSE'],
                        default='sense', help='Type of unit: sense, lemma, sentence, argument structure (abstract, medium or concrete roles)')


    parser.add_argument('data_type',  nargs='+', choices=['feats', 'cons', 'pats'],
                        help='Type of data of the output (prob, bin)')

    parser.add_argument('-o', '--output', default='../GeneratedData/aspect2/',
                        help='Output folder for the training/text csv files')

    parser.add_argument('-itest', '--inputTest', default='../RawData/SensemTestTrain/',
                        help='Input folder for the sensem test part')

    parser.add_argument('-itrain', '--inputTrain', default='../RawData/SensemTestTrain/',
                        help='Input folder for the sensem train part')

    parser.add_argument('-corpus', '--corpus_file', default='../RawData/sensemMin2.xml', help='Corpus sensem')

    parser.add_argument('-clusters', '--path_clusters', default='../Auxdata/ToUse/bueno/',
                        help='Folder that contains the different WE to use. Mandatory for "syntaxSP", "morfSPref", "morfoSPrepSPref"')

    args = parser.parse_args()
    print(args)
    # main('aspect', 'lemma', 'synCat', 'prob', 'sense', 'feats', 'prueba/', '/home/lara/Documents/CODIGO/clustering/RawData/SensemTestTrain/testVistos5f.xml', '/home/lara/Documents/CODIGO/clustering/RawData/SensemTestTrain/testVistos5f.xml','/home/lara/Documents/CODIGO/clustering/RawData/SensemTestTrain/testVistos5f.xml','/home/lara/Documents/CODIGO/clustering/Auxdata/ToUse/bueno/periodico_100D_w5_min5_100cl.txt')

    # set folders & files
    outTraining = os.path.join(args.output + 'training/')
    if not os.path.exists(outTraining):
        os.makedirs(outTraining)

    outTestNV = os.path.join(args.output + 'testNV/')
    if not os.path.exists(outTestNV):
        os.makedirs(outTestNV)

    corpus = Corpus(args.corpus_file, args.inputTrain, args.inputTest, args.unit)
    corpus.get_train_test_data()
    corpus.get_sentences()

    if args.supra_arg:
        if 'aspect' in args.supra_arg:
            corpus.aspect = True
        if 'perif' in args.supra_arg:
            corpus.periph = True
        if 'aspectual' in args.supra_arg:
            corpus.aspectuality = True
        if 'modal' in args.supra_arg:
            corpus.modality = True
        if 'polar' in args.supra_arg:
            corpus.polarity = True
        if 'const' in args.supra_arg:
            corpus.construction = True
        corpus.get_sentence_features()


    corpus.semantic_info = args.semantic_info
    corpus.syntactic_info = args.syntactic_info
    corpus.concurrences_format = args.concurrence_type
    pickled = open(args.path_clusters, 'rb')
    corpus.clusters = pickle.load(pickled, encoding='bytes')
    corpus.data_type = args.data_type
    corpus.get_argument_features()
    corpus.convert_to_matrix(args.output)



if __name__ == '__main__':
    main()

