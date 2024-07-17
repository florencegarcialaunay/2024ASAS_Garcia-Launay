from math import log as ln
from random import randint
from copy import deepcopy
import numpy as np
from math import exp
from csv import *
import pandas as pd
from math import exp

class Batch:

    def __init__(self, nb_pigs, ID=' '):
        self.ID = ID  # batch ID
        self.dict_pigs = {}  # dictionnary of pigs objects in the batch - keys are pigs' names
        self.nbpigs = nb_pigs # number of pigs in the batch
        self.AverageLW = 0 # average liveweight of the pigs in the batch
        self.Fattening_time = 0  # days - time since the start of fattening
        self.ready_pigs = [] # list of pigs ready for delivery to slaughterhouse

        for i in range(self.nbpigs):  # to create pigs objects in the batch
            temp = self.ID + 'P' + str(i)
            ADG_pot = np.random.normal(0.850, 0.125, 1)[0] # (mean, sd, number)
            Agei = round(np.random.normal(70.0, 2.5, 1)[0])
            BWi = np.random.normal(32.0, 2.5, 1)[0]
            BGompertz = np.random.normal(0.015, 0.00011, 1)
            self.dict_pigs[temp] = Pig(ADG_pot, Agei, BWi, BGompertz, temp)
            self.AverageLW += self.dict_pigs[temp].LW

        self.AverageLW /= self.nbpigs
        self.IDpigs = list(self.dict_pigs.keys())  # list of pigs' names
        self.IDpigs.sort()

    def batch_growth(self, param_duration, file_day, i):
        listpigs = deepcopy(self.IDpigs)
        self.AverageLW = sum([self.dict_pigs[i].LW for i in self.IDpigs])/len(listpigs)      
        for name in listpigs:
            self.dict_pigs[name].function_growth(param_duration)
            p = self.dict_pigs[name]
            print(p.ID + ' ' + str(p.Age) + ' ' + '%.*f' % (3, p.LW) + ' ' + '%.*f' % (3, p.DG), file = file_day)
            file_day.flush()

    
    def delivery(self, param_duration, file_perf, i):

        list_delivery = []
        
        # listing the names of the pigs ready to be delivered to slaughterhouse
        for name in self.IDpigs:
            if (self.dict_pigs[name].LW >= param_duration) & (name not in self.ready_pigs): #target slaughter weight 120 kg
                self.ready_pigs.append(name)
                self.dict_pigs[name].delivery = "slaughterhouse"
                list_delivery.append(name)

        # if the maximum fattening duration is reached, all pigs will be sent
        if i == param_duration:
            list_delivery = self.IDpigs
            for name in list_delivery:
                if self.dict_pigs[name].delivery != "slaughterhouse":
                    self.dict_pigs[name].delivery = "Room_emptying"

        # pigs are sent, removed from lists and output file filled
        for name in list_delivery:
            p = self.dict_pigs[name]
            print(p.ID + ' ' + p.delivery + ' ' + str(p.Agei) + ' ' + str(p.Age) + ' ' + '%.*f' % (3, p.LWi) + ' ' + '%.*f' % (3, p.LW)
                  + ' ' + '%.*f' % (3, ((p.LW - p.LWi)/(p.Age - p.Agei))) , file=file_perf)
            file_perf.flush()
          
            self.IDpigs.remove(name)
            if name in self.ready_pigs:
                self.ready_pigs.remove(name)
            del self.dict_pigs[name]

        
class Pig:

    def __init__(self, ADG_pot, Agei, BWi, BGompertz, ID=' '):

        self.ADGmoy = ADG_pot # g/d - Average daily protein gainépôt protéique moyen journalier (en g/j)
        self.ID = ID  # Id number of the pig        
        self.LWi = BWi  # kg - Initial LW
        self.LW = self.LWi  # kg - LW
        self.Agei = Agei  # days - Initial age
        self.Age = self.Agei  # days - Age
        self.DG = 0 # kg/d - daily gain
        self.BGompertz = BGompertz # precocity parameter
        self.delivery = 'no' # initalising the status for delivery to slaughterhouse

    def function_growth(self, param_duration):

        AgeFin = round((param_duration - self.LWi) / (self.ADGmoy)) + self.Agei
        BWFin = (AgeFin - self.Agei) * (self.ADGmoy) + self.LWi;
        Gompertz = exp(-self.BGompertz * (AgeFin - self.Agei))
        self.BWmat = BWFin * ((BWFin / self.LWi) ** (Gompertz / (1 - Gompertz)))
        self.DG = self.BGompertz * self.LW * ln(self.BWmat / self.LW) # potential for growth g/d
        self.LW += self.DG
        self.Age +=1


                   