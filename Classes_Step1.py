from math import log as ln
from random import randint
from copy import deepcopy
import numpy as np
from math import exp

class Batch:

    def __init__(self, profile, Variability, nb_pigs, ruleseq, Ageimoy, BWimoy, dict_INRAPorc, ID=' '):
    
        self.ID = ID  # batch ID
        self.dict_pigs = {}  # dictionnary of pigs objects in the batch - keys are pigs' names
        self.nbpigs = nb_pigs # number of pigs in the batch
        self.AverageLW = 0 # average liveweight of the pigs in the batch
        self.Fattening_time = 0  # days - time since the start of fattening
        self.stocknbpigs = [self.nbpigs]  # list of the number of pigs each day
        self.profile = profile  # either one profile or a dictionnary of pig profiles
        self.ready_pigs = [] # list of pigs ready for delivery to slaughterhouse
        nompro = list(self.profile.keys())
        nompro.sort()    
        if Variability == 0: # in this case, all pigs have the same growth and intake profile
            for i in range(self.nbpigs):  # to create pigs objects in the batch
                temp = self.ID + 'P' + str(i)
                if i < (self.nbpigs/2):
                    self.dict_pigs[temp] = Pig(self.profile[nompro[0]], Ageimoy, BWimoy, dict_INRAPorc, temp)
                else:
                    self.dict_pigs[temp] = Pig(self.profile[nompro[1]], Ageimoy, BWimoy, dict_INRAPorc, temp)
                self.AverageLW += self.dict_pigs[temp].LW

        elif Variability == 1: # in this case, all pigs have a different growth and intake profile
            n = 0
            for i in nompro:  # to create pigs objects in the batch
                temp = self.ID + 'P' + str(n)
                n += 1
                self.dict_pigs[temp] = Pig(self.profile[i], Ageimoy, BWimoy, dict_INRAPorc, temp)
                self.AverageLW += self.dict_pigs[temp].LW
        self.AverageLW /= self.nbpigs
        self.IDpigs = list(self.dict_pigs.keys())  # list of pigs' names
        self.IDpigs.sort()
        
        # Initializing feed sequence plan
        self.seq = deepcopy(ruleseq)  # Feed sequence plan 
        self.numseq = 0  # Number of the feeding phase (0 corresponds to first phase)
        self.dureeseq = 0  # Number of days since the start of the current feeding phase

    def batch_growth(self, Feeds, dict_INRAPorc, param, function_INRAPorc, file_day, i):
        listpigs = deepcopy(self.IDpigs)
        self.AverageLW = sum([self.dict_pigs[i].LW for i in self.IDpigs])/len(listpigs)
        ModeFin = self.seq[self.numseq][0]
        ValFin = self.seq[self.numseq][1]  # on stocke la valeur finale associée
        if (ModeFin ==2) & (self.AverageLW >= ValFin):# if the average live weight of the batch is higher than the rule to move to next feeding phase
            self.numseq +=1
            self.dureeseq = 0
        else:
            self.dureeseq +=1
        Current_Feed1 = Feeds.dict_feeds[self.seq[self.numseq][2]]
        Current_Feed2 = Feeds.dict_feeds[self.seq[self.numseq][3]]
        Percentage_feed1 = self.seq[self.numseq][4]
        
        for name in listpigs:
            #self.dict_pigs[name].growth(self.AverageLW, Current_Feed, dict_INRAPorc, param)
            self.dict_pigs[name] = function_INRAPorc(self.dict_pigs[name], Current_Feed1, Current_Feed2, Percentage_feed1, dict_INRAPorc, param)
            self.dict_pigs[name].Age +=1
            p = self.dict_pigs[name]
            print(param['scenario'] + ' ' + p.ID + ' ' + p.name + ' '+ str(i) + ' ' + str(p.Age) + ' ' + '%.*f' % (3, p.LW) + ' ' + '%.*f' % (3, p.DG)
                  + ' ' + '%.*f' % (3, p.Prot) + ' ' + '%.*f' % (3, p.Lip) + ' ' + '%.*f' % (3, p.Feed_Intake)
                  + ' ' + '%.*f' % (3, p.FecalN) + ' ' + '%.*f' % (3, p.UrinN), file = file_day)
            file_day.flush()


    def delivery(self, Feeds, param, file_perf, i):

        list_delivery = []
        
        # listing the names of the pigs ready to be delivered to slaughterhouse
        for name in self.IDpigs:
            if (self.dict_pigs[name].LW >= param['target_slaughter_weight']) & (name not in self.ready_pigs):
                self.ready_pigs.append(name)
                self.dict_pigs[name].delivery = "slaughterhouse"
                list_delivery.append(name)

        # if the maximum fattening duration is reached, all pigs will be sent
        if i == param['duration']:
            list_delivery = self.IDpigs
            for name in list_delivery:
                if self.dict_pigs[name].delivery != "slaughterhouse":
                    self.dict_pigs[name].delivery = "Room_emptying"

        # pigs are sent, removed from lists and output file filled
        for name in list_delivery:
            p = self.dict_pigs[name]
            string_feed = ''
            for feed in p.dict_stockFeed.keys():
                string_feed = string_feed + '\t' + str(feed) + '\t' + '%.*f' % (3, p.dict_stockFeed[feed])
            print(param['scenario'] 
                  + '\t' + p.ID 
                  + '\t' + p.name 
                  + '\t' + p.delivery 
                  + '\t' + str(p.Agei) 
                  + '\t' + str(p.Age) 
                  + '\t' + '%.*f' % (3, p.LWi) 
                  + '\t' + '%.*f' % (3, p.LW)
                  + '\t' + '%.*f' % (3, p.ProtInit) 
                  + '\t' + '%.*f' % (3, p.Prot) 
                  + '\t' + '%.*f' % (3, p.LipInit) 
                  + '\t'  + '%.*f' % (3, p.Lip)
                  + '\t' + '%.*f' % (3, p.Cumulated_FM_intake) 
                  + '\t' + '%.*f' % (3, p.N_intake) 
                  + '\t' + '%.*f' % (3, p.N_excreted)
                  + '\t' + '%.*f' % (3, p.TAN) 
                  + '\t' + '%.*f' % (3, ((p.LW - p.LWi)/(p.Age - p.Agei))) 
                  + '\t' + '%.*f' % (3, (p.Cumulated_FM_intake/(p.LW - p.LWi)))
                  + string_feed, file=file_perf)
            file_perf.flush()
          
            self.IDpigs.remove(name)
            if name in self.ready_pigs:
                self.ready_pigs.remove(name)
            del self.dict_pigs[name]

        
class Pig:

    def __init__(self, profile, Ageimoy, BWimoy, dict_INRAPorc, ID=' '):

        # INRAPorc profile

        self.PDMoy = profile['PDMoy'] # g/d - Average daily protein gainépôt protéique moyen journalier (en g/j) 
        self.BGompertz = profile['BGompertz']  # Precocity parameter in Gompterz function for potential protein deposition
        self.Maint = profile['Maint']  # Maintenance factor
        self.PVPDmax = profile['PVmr2']  # kg - BW at which the pig can reach its potential maximum protein deposition
        self.Carcass_Yield = profile['Carcass']  # Carcass percentage % of BW
        self.a_eq = profile['a']  # Paramter in the intake equation
        self.b_eq = profile['b']  # Parameter in the intake equation
        self.sex = ' '  # Sexe du cochon : femelle, mâle entier, mâle castré
        self.delivery = 'no' # initalising the status for delivery to slaughterhouse
        sexporc = profile['sex']
        if profile['sex'] == 0:
            self.sex = 'F' # female
        elif profile['sex'] == 1:
            self.sex = 'CM' # castrated male
        elif profile['sex'] == 2:
            self.sex = 'EM' # entire male
        else: # on average 50% females, 50% entire males
            r = randint(1, 2)
            if r == 1:
                self.sex = 'F'
            else:
                self.sex = 'EM'

        # LW and body composition initializing
        self.name = profile['name'] # name of the INRAPorc profile
        self.ID = ID  # Id number of the pig        
        self.LWi = profile['BWi'] * BWimoy / 30.0  # kg - Initial LW
        self.LWiseq = self.LWi  # kg - Initial LW at the start of the feeding phase
        self.LW = self.LWi  # kg - LW
        self.Agei = int(profile['Agei'] * Ageimoy / 70)  # days - Initial age
        self.Ageiseq = self.Agei  # days - Initial Age at the start of the feeding phase
        self.Age = self.Agei  # days - Age
        self.DG = 0 # kg/d - daily gain
        self.ProtInit = self.CalcProt(self.LWi)  # profilporc['ProtInit']    # kg - initial body protein
        self.Prot = self.CalcProt(self.LW)  # profilporc['ProtInit']    # kg - body protein
        self.LipInit = self.CalcLipProt(self.LWi, self.ProtInit, dict_INRAPorc)  # kg - initial body lipids
        self.Lip = self.CalcLipProt(self.LW, self.Prot, dict_INRAPorc)   # kg - body lipids
        self.dict_stockFeed = {}  # kg - Total feed consumption for each feed
        self.FirstAALimit = 'none' # first AA limiting protein deposition
        self.OMexc = 0 # excreted organic matter - kg
        self.dFibre = 0 # digested fibre - kg - for further calculation of enteric methane
        self.N_intake = 0 # total nitrogen intake - kg
        self.N_excreted = 0 # total N excreted - kg
        self.TAN = 0 # total ammoniacal N excreted - kg
        self.Cumulated_FM_intake = 0 # cumulated feed intake - kg FM
        self.Feed_intake = 0 # instantaneous daily feed intake - kg FM
        self.FecalN = 0 # fecal nitrogen kg/d
        self.UrinN = 0 # urinary nitrogen kg/d
        self.RetainedN = 0 # retained nitrogen into body mass kg/d
        self.FecalP = 0 # fecal phosphorus kg/d
        self.UrinP = 0 # urinary phosphorus kg/d
        self.RetainedP = 0 # retained phosphorus into body mass kg/d


    def CalcEBW(self, BW):
        # allometric function to calculate empty body weight according to BW
        aEBW = 0.8901791
        bEBW = 1.0142019
        return aEBW * (BW ** bEBW)

    def CalcBW(self, EBW):
        # allometric function to calculate empty body weight according to BW
        aEBW = 0.8901791
        bEBW = 1.0142019
        return (EBW / aEBW) ** (1 / bEBW)

    def CalcProt(self, BW):
        # function to calculate body protein according to BW
        return self.CalcEBW(BW) * 0.16

    def CalcLipProt(self, BW, prot, dict_INRAPorc):
        # function to calculate body lipids according to BW and body protein
        if prot > 0:
            x = (1000 * self.CalcEBW(BW) - dict_INRAPorc['Pallom'] * exp(dict_INRAPorc['Ballom'] * ln(1000 * prot))) / dict_INRAPorc['Lallom']
            if x > 0:
                return exp(-(-ln(x) + dict_INRAPorc['Ballom'] * ln(1000)) / dict_INRAPorc['Ballom'])
            else:
                return 0
        else:
            return 0

    def FeedIntake(self, BW, a, b):
        return (a * (b * BW * exp(-b * BW)) + 1) * 0.75 * (BW ** 0.6) # net energy intake in MJ/d - Gamma function of maintenance 
                   
       
class Feed:

    def __init__(self, dict_Feed):
        self.name = dict_Feed['name']
        self.CP = dict_Feed['CP'] # g/kg feed - crude protein content
        self.starch = dict_Feed['starch'] # g/kg feed - starch content
        self.sugars = dict_Feed['sugars'] # g/kg feed - sugars content
        self.dig_lipids = dict_Feed['dlipids'] # g/kg feed - digestible lipids content
        self.dig_CP = dict_Feed['dCP']  # g/kg dig crude protein content for growing pigs
        self.SIDN = dict_Feed['SIDN'] # g/kg standardized ileal digestible N
        self.NE = dict_Feed['NE'] # MJ/kg feed - net energy content
        self.SIDLys = dict_Feed['SIDLys'] # g/kg feed - standardized ileal digestible lysine
        self.DM = dict_Feed['DM'] # dry matter content - g/kg feed
        self.OM = dict_Feed['OM'] # organic matter content - g/kg feed
        self.dOM = dict_Feed['dOM'] # digestible organic matter - g/kg feed
        self.dFibre = dict_Feed['dFibre'] # digestible residu - g/kg feed as digestible OM minus digestible protein, fat, starch, sugar
        self.dCF = dict_Feed['dCF'] # digestible crude fibre - g/kg feed

        
class Feed_storage:

    def __init__(self, dict_feeds_storage):
        self.dict_feeds = {}
        for key in dict_feeds_storage.keys():
            self.dict_feeds[dict_feeds_storage[key]['name']] = Feed(dict_feeds_storage[key])



