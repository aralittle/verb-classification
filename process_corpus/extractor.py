from lxml import etree as ET
from collections import Counter
import pandas

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

prep = ['a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en' , 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'según'.decode('utf-8'), 'sin', 'so', 'sobre', 'tras', 'versus ', 'vía', 'vía'.decode('utf-8')]


class Argument:

    def __init__(self, argument):
        self.argument = argument
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
                        clusterID = clusters[w.text.lower().decode('utf-8')]
                        clustersArgument.append(str(clusterID))
                    except:
                        clustersArgument.append('OOS')  # word was not in embeddings

            argumentInfo = '*'.join(clustersArgument)
            self.synFunctSelectPref = self.argument.attrib["fs"]+'*'+argumentInfo.encode('utf-8')

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
                if preposition == 'al':
                    prepoInfo = 'a'
                if preposition == 'del':
                    prepoInfo = 'de'
            self.synCatPrep = self.argument.attrib["cat"] + '*' + prepoInfo.encode('utf-8')

            clustersArgument = []
            if self.argument.attrib["cat"] in ['NP', 'InfSC']:
                words = self.argument.findall('.//word[@core]')
                for w in words:
                    try:  # retrieve cluster ID of word
                        clusterID = clusters[w.text.lower().decode('utf-8')]
                        clustersArgument.append(str(clusterID))
                    except:
                        clustersArgument.append('OOS')  # word was not in embeddings
            argumentInfo = '*'.join(clustersArgument)
            self.synCatCluster = self.argument.attrib["fs"] + '*' + argumentInfo.encode('utf-8')

            self.synCatPrepCluster = self.synCatPrep + argumentInfo

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
    def __init__(self, sentence):
        self.sentence = sentence
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
        if not self.arguments:
            self.get_sentence_info()

        if info_type == 'sentido':
            key = self.lemma + '_' + self.sense

        if info_type == 'frase':
            self.key = self.lemma + '_' + self.sense + '_' + self.ide

        RolesAbs = []
        RolesMed = []
        RolesSpec = []
        for arg in self.arguments:
            argument = Argument(arg)
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
    def __init__(self, info_type):

        self.sentences = []
        self.info_type = info_type
        self.syntactic_info = ''
        self.semantic_info = ''
        self.aspect = False
        self.periph = False
        self.aspectuality = False
        self.modality = False
        self.polarity = False
        self.construction = False

        self.dicSentenceFeats = {}
        self.dicFeats = {}
        self.dicCons = {}
        self.dicPattern = {}

    def get_sentences(self, doc):
        root = ET.parse(doc)
        sentences = root.iter('sentence')
        for s in sentences:
            sentence = Sentence(s)
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

                argInfo = []

                if self.semantic_info:
                    arg.ontological_info()

                    if self.semantic_info == 'TCO':
                        self.dicFeats[sentence.key].update([arg.ontoTCO])
                        argInfo.append(arg.ontoTCO)

                    if self.semantic_info == 'TCOsplit':
                        for feat in arg. ontoTCOSplit:
                            self.dicFeats[sentence.key].update([feat])
                            argInfo.append(feat)

                    if self.semantic_info == 'supersense':
                        self.dicFeats[sentence.key].update([arg.ontoSupersense])
                        argInfo.append(arg.ontoSupersense)

                    if self.semantic_info == 'sumo':
                        self.dicFeats[sentence.key].update([arg.ontoSumo])
                        argInfo.append(arg.ontoSumo)

                    if self.semantic_info == 'lemma':
                        self.dicFeats[sentence.key].update([arg.lemma])
                        argInfo.append(arg.lemma)

                    if self.semantic_info == 'semsem':
                        self.dicFeats[sentence.key].update([arg.sensemRole])
                        argInfo.append(arg.sensemRole)

                    if self.semantic_info == 'medium':
                        self.dicFeats[sentence.key].update([arg.mediumRole])
                        argInfo.append(arg.mediumRole)

                    if self.semantic_info == 'abstract':
                        self.dicFeats[sentence.key].update([arg.abstractRole])
                        argInfo.append(arg.abstractRole)

                if self.syntactic_info:
                    arg.get_syntactic_function_info()

                    if 'synFunct' in self.syntactic_info:
                        if self.syntactic_info == 'synFunct':
                            self.dicFeats[sentence.key].update([arg.synFunct])
                            argInfo.append(arg.synFunct)

                        if self.syntactic_info == 'synFunctSelectPref':
                            self.dicFeats[sentence.key].update([arg.synFunctSelectPref])
                            argInfo.append(arg.synFunctSelectPref)

                    if 'synCat' in self.syntactic_info:
                        arg.get_syntactic_category_info(prep)
                        if self.syntactic_info == 'synCat':
                            self.dicFeats[sentence.key].update([arg.synCat])
                            argInfo.append(arg.synCat)

                        if self.syntactic_info == 'synCatPrep':
                            self.dicFeats[sentence.key].update([arg.synCatPrep])
                            argInfo.append(arg.synCatPrep)

                        if self.syntactic_info == 'synCatCluster':
                            self.dicFeats[sentence.key].update([arg.synCatCluster])
                            argInfo.append(arg.synCatCluster)

                        if self.syntactic_info == 'synCatPrepCluster':
                            self.dicFeats[sentence.key].update([arg.synCatPrepCluster])
                            argInfo.append(arg.synCatPrepCluster)

                cons = '+'.join(argInfo)
                self.dicCons[sentence.key].update([cons])
                sentenceInfo.append(cons)
            pat = '+'.join(sentenceInfo)
            self.dicPattern[sentence.key].update([pat])

    def convert_to_matrix(self, concurrences_format, dataType, output_path):
        data_dic = {}
        if dataType == 'feats':
            data_dic = self.dicFeats
        if dataType == 'cons':
            data_dic = self.dicCons
        if dataType == 'pats':
            data_dic = self.dicPattern

        for key in self.dicSentenceFeats:
            if key in data_dic:
                for val in self.dicSentenceFeats:
                    data_dic[key].update([val])
            else:
                data_dic[key] = self.dicSentenceFeats[key]

        df = pandas.DataFrame(data_dic).transpose()
        if concurrences_format == 'bin':
            df = df.round(1)
        else:
            df = df.div(df.sum(axis=1), axis=0)
        df.to_csv(output_path)






