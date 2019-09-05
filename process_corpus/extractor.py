
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
        self.ontoTCO = ''
        self.ontoTCOSplit = []
        self.ontoSupersense = ''
        self.ontoSumo = ''
        self.lemma = ''

    def get_roles(self, argument):
        try:
            self.abstractRole = role_mapping[argument.attrib['rs']].split('-')[0]+'_SS'
            self.mediumRole = '-'.join(role_mapping[argument.attrib['rs']].split('-')[:2])+'_SS'
            self.sensemRole = argument.attrib['rs']+'_SS'
        except KeyError:
            print('argument does not have role tag')

    def get_syntactic_function_info(self, argument):
        try:
            self.synFunct = argument.attrib["fs"]+'_SS'

            clustersArgument = []
            if argument.attrib["fs"] in ['Subject', "Direct obj.", "Indirect obj."]:
                words = argument.findall('.//word[@core]')
                for w in words:
                    try:  # retrieve cluster ID of word
                        clusterID = clusters[w.text.lower().decode('utf-8')]
                        clustersArgument.append(str(clusterID))
                    except:
                        clustersArgument.append('OOS')  # word was not in embeddings

            argumentInfo = '*'.join(clustersArgument)
            self.synFunctSelectPref = argument.attrib["fs"]+'*'+argumentInfo.encode('utf-8')

        except KeyError:
            print('argument does not have syntactic function tag')

    def get_syntactic_category_info(self, argument, prep):
        try:
            self.synCat = argument.attrib["cat"]+'_SS'

            prepoInfo = ''
            if self.synCat[:2] == 'PP':
                subwords = argument.findall('.//word')
                preposition = subwords[0].text.lower()  # .decode('UTF-8')
                if preposition.split('_')[0] in prep:
                    prepoInfo = preposition.split('_')[0]
                if preposition == 'al':
                    prepoInfo = 'a'
                if preposition == 'del':
                    prepoInfo = 'de'
            self.synCatPrep = argument.attrib["cat"] + '*' + prepoInfo.encode('utf-8')

            clustersArgument = []
            if argument.attrib["cat"] in ['NP', 'InfSC']:
                words = argument.findall('.//word[@core]')
                for w in words:
                    try:  # retrieve cluster ID of word
                        clusterID = clusters[w.text.lower().decode('utf-8')]
                        clustersArgument.append(str(clusterID))
                    except:
                        clustersArgument.append('OOS')  # word was not in embeddings
            argumentInfo = '*'.join(clustersArgument)
            self.synCatCluster = argument.attrib["fs"] + '*' + argumentInfo.encode('utf-8')

            self.synCatPrepCluster = self.synCatPrep + argumentInfo

        except KeyError:
            print('argument does not have syntactic category tag')

    def ontological_info(self, argument):
        # retrieve ontological category of words
        wordsS = argument.findall('.//word[@core]')
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

