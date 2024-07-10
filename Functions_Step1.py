from csv import *
from pathlib import Path
import pandas as pd
import numpy as np
from math import exp
from math import log as ln

def read_param(path):

    input_file = open(path, "r")
    c = reader(input_file, delimiter=';')
    next(c)  # Saute la première ligne (les headers)
    global param
    param = {}  # Dictionnaire des variables d'entrées. La clé est le nom de la variable.
    for row in c:  # On parcourt toutes les lignes du fichiers. Chaque ligne contient une variable
        n = int(row[1])  # Taille de la variable : nombre de lignes
        m = int(row[2])  # Taille de la variable : nombre de colonnes
        clef = row[0]  # Chaine de caractère correspondant au nom de la variable (clef du dictionnaire param)
        if row[
            4] != 'str':  # Si la variable lue n'est pas de type chaine de caractère (si c'est un entier ou un nombre flottant (float))
            if n == 1:  # Si la variable lue a une seule ligne
                if m == 1:  # Si la variable lue a une seule colonne
                    ch1 = row[4] + '(' + row[
                        3] + ')'  # On stocke la valeur de la variable sous forme de chaine de caractère
                    param[clef] = eval(ch1)
                else:  # Si la variable lue a plusieurs colonnes
                    param[clef] = []  # On stocke la variable sous forme d'un vecteur de chaines de caractère
                    ch2 = row[3].split()
                    for i in range(m):
                        ch1 = eval(row[4] + '(' + ch2[i] + ')')
                        param[clef].append(ch1)
            else:  # Si la variable lue a plusieurs lignes
                ch2 = row[3].split()  # On stocke la variable sous forme d'une matrice de chaines de caractère
                param[clef] = np.zeros((n, m))  # Crée la matrice nulle
                for i in range(n):
                    for j in range(m):
                        param[clef][i, j] = eval(row[4] + '(' + ch2[m * i + j] + ')')
        else:  # Si la variable lue est de type chaine de caractère
            if n == 1:  # Si la variable lue a une seule ligne
                if m == 1:  # Si la variable lue a une seule colonne
                    param[clef] = row[3]  # On stocke la valeur de la variable sous forme de chaine de caractère
                else:  # Si la variable lue a plusieurs colonnes
                    param[clef] = []  # On stocke la variable sous forme d'un vecteur de chaines de caractère
                    ch2 = row[3].split()
                    for i in range(m):
                        param[clef].append(ch2[i])
            else:  # Si la variable lue a plusieurs lignes
                ch2 = row[3].split()  # On stocke la variable sous forme d'une matrice de chaines de caractère
                param[clef] = np.eye(n, m)
                for i in range(n):
                    for j in range(m):
                        param[clef][i, j] = ch2[m * i + j]
    input_file.close()  # On referme le flux permettant de lire le fichier d'entrée
    return param

def lect_profile(file):
    
    file = Path(file)
    f = open(file, 'r')
    out = pd.read_csv(f, sep=';', )
    dict_profile = {}
    dict_profilanim = out.to_dict('records')
    for i in range(0, len(dict_profilanim)):
        a = {'num': i + 1}
        dict_profilanim[i].update(a)
        dict_profilanim[i]['name'] = dict_profilanim[i].pop('Nom')
        dict_profilanim[i]['sex'] = dict_profilanim[i].pop('Sexe')
        dict_profilanim[i]['BGompertz'] = dict_profilanim[i].pop('Bgompertz')
        dict_profilanim[i]['Maint'] = dict_profilanim[i].pop('Entretien')
        dict_profilanim[i]['Carcass'] = dict_profilanim[i].pop('Carcasse')
        dict_profilanim[i]['PVmr2'] = float(dict_profilanim[i]['PVmr2'])
        dict_profilanim[i]['BWi'] = dict_profilanim[i].pop('PVInit')
        dict_profilanim[i]['Agei'] = dict_profilanim[i].pop('AgeInit')        
        del dict_profilanim[i]['ModeFin']
        del dict_profilanim[i]['Duree']
        dict_profile[dict_profilanim[i].get("name")] = dict_profilanim[i]
    f.close()
    ProfilAnim = dict_profile
    return ProfilAnim

def lect_feed(file):
    
    file = Path(file)
    f = open(file, 'r')
    out = pd.read_csv(f, sep=';', )
    dict_feeds ={}
    dict_feedsfromfile = out.to_dict('records')
    for i in range(0, len(dict_feedsfromfile)):
        a = {'num': i + 1}
        dict_feedsfromfile[i].update(a)
        dict_feedsfromfile[i]['name'] = dict_feedsfromfile[i].pop('Name')
        dict_feedsfromfile[i]['starch'] = dict_feedsfromfile[i].pop('Starch')
        dict_feedsfromfile[i]['sugars'] = dict_feedsfromfile[i].pop('Sugars')
        dict_feedsfromfile[i]['dlipids'] = dict_feedsfromfile[i].pop('Dig. fat G')
        dict_feedsfromfile[i]['dCP'] = dict_feedsfromfile[i].pop('Dig. CP G')
        dict_feedsfromfile[i]['CP'] = dict_feedsfromfile[i].pop('CP')
        dict_feedsfromfile[i]['SIDN'] = dict_feedsfromfile[i].pop('Dig. N')
        dict_feedsfromfile[i]['NE'] = float(dict_feedsfromfile[i]['NE G'])
        dict_feedsfromfile[i]['SIDLys'] = dict_feedsfromfile[i].pop('Dig. Lys')
        dict_feedsfromfile[i]['OM'] = dict_feedsfromfile[i].pop('OM')
        dict_feedsfromfile[i]['DM'] = dict_feedsfromfile[i].pop('DM')
        dict_feedsfromfile[i]['dOM'] = dict_feedsfromfile[i].pop('Dig. OM G')
        dict_feedsfromfile[i]['dFibre'] = dict_feedsfromfile[i].pop('Dig. residue G') # digested fibre - loss of energy as CH4 is dFibre x 670 J/g for pigs - Noblet et al. 2004
        dict_feedsfromfile[i]['dCF'] = dict_feedsfromfile[i].pop('dig. CF G') # digestesd crude fibre 
        dict_feeds[dict_feedsfromfile[i].get("name")] = dict_feedsfromfile[i]
    
    f.close()
    return dict_feeds
    
def function_INRAPorc(OnePig, Feed1, Feed2, Pctfeed1, dict_INRAPorc, param):
    NE_intake = OnePig.FeedIntake(OnePig.LW, OnePig.a_eq, OnePig.b_eq) # intake in net energy - MJ/d
    FM_intake = NE_intake / (Feed1.NE * Pctfeed1/100 + Feed2.NE * (1-Pctfeed1/100)) # feed intake in kg fresh matter / d
    OnePig.Feed_Intake = FM_intake
    DM_intake = FM_intake * ((Feed1.DM/1000) * Pctfeed1/100 + (Feed2.DM/1000) * (1-Pctfeed1/100))# dry matter intake kg/d
    list_name = list(set([Feed1.name, Feed2.name]))
    for f in list_name:
        if f == Feed1.name :
            pct = Pctfeed1
        else: pct = 100 - Pctfeed1
        if f in OnePig.dict_stockFeed.keys():
            OnePig.dict_stockFeed[f] += FM_intake * pct/100
        else:
            OnePig.dict_stockFeed[f] = FM_intake * pct/100

    TabDEIntake = np.zeros(6)
    TabDEIntake[0] = FM_intake * (Feed1.dig_CP * Pctfeed1/100 + Feed2.dig_CP * (1-Pctfeed1/100)) # digestible CP intake - g
    TabDEIntake[1] = FM_intake * (Feed1.dig_lipids * Pctfeed1/100 + Feed2.dig_lipids * (1-Pctfeed1/100))  # digestible Fat intake - g
    TabDEIntake[2] = FM_intake * (Feed1.starch * Pctfeed1/100 + Feed2.starch * (1-Pctfeed1/100))  # digestible starch intake (digestibility = 100%) - g
    TabDEIntake[3] = FM_intake * (Feed1.sugars * Pctfeed1/100 + Feed2.sugars * (1-Pctfeed1/100))  # digestible sugars intake (digestibility = 100%) - g
    TabDEIntake[4] = FM_intake * (Feed1.dCF * Pctfeed1/100 + Feed2.dCF * (1-Pctfeed1/100))  # # digested crude fibre intake - g
    TabDEIntake[5] = FM_intake * (Feed1.dFibre * Pctfeed1/100 + Feed2.dFibre * (1-Pctfeed1/100))  # digested residu intake - g

    TabAAdig = np.zeros(2)
    TabAAm = np.zeros(2)
    TabAAfG = np.zeros(2)
    TabPDAA = np.zeros(2)
    TabAAdig[0] = FM_intake * (Feed1.SIDN * Pctfeed1/100 + Feed2.SIDN * (1-Pctfeed1/100)) # N - g/day
   
    TabAAdig[1] = FM_intake * (Feed1.SIDLys * Pctfeed1/100 + Feed2.SIDLys * (1-Pctfeed1/100)) # lysine - g/day
     
    for i in range(2):
        TabAAm[i] = dict_INRAPorc['AAm75'][i] * (OnePig.LW ** 0.75) + DM_intake * dict_INRAPorc['AAendo'][i]
        TabAAfG[i] = (TabAAdig[i] - TabAAm[i]) * dict_INRAPorc['kAA'][i]  # Acides aminés disponibles pour la croissance (for Growth) par jour
        TabPDAA[i] = TabAAfG[i] / dict_INRAPorc['AAbody'][i]
    OnePig.FirstAALimit = 'N'
    PDFirstLimit = TabPDAA[0]
    if TabPDAA[1] < PDFirstLimit:
        PDFirstLimit = TabPDAA[1]
        OnePig.FirstAALimit = 'Lys'
    NEintakeAL = 1000 * NE_intake # intake in net energy - kJ/d
    AgeFin = round((OnePig.CalcProt(param['target_slaughter_weight']) - OnePig.ProtInit) / (OnePig.PDMoy / 1000)) + OnePig.Agei
    ProtFin = (AgeFin - OnePig.Agei) * (OnePig.PDMoy / 1000) + OnePig.ProtInit;
    Gompertz = exp(-OnePig.BGompertz * (AgeFin - OnePig.Agei))
    OnePig.Pmat = ProtFin * ((ProtFin / OnePig.ProtInit) ** (Gompertz / (1 - Gompertz)))
    PDAL = 1000 * OnePig.BGompertz * OnePig.Prot * ln(OnePig.Pmat / OnePig.Prot) # potential for protein deposition g/d
    FHP60 = dict_INRAPorc['FHPint'] + dict_INRAPorc['FHPpente'] * FM_intake * 1000 * (Feed1.NE * Pctfeed1/100 + Feed2.NE * (1-Pctfeed1/100)) / (OnePig.LW ** 0.6)
    NEmAL = OnePig.Maint * (dict_INRAPorc['Standing'] * dict_INRAPorc['NEact60h'] + dict_INRAPorc['kBR'] * FHP60) * (OnePig.LW ** 0.6)
    NEm = OnePig.Maint * (dict_INRAPorc['Standing'] * dict_INRAPorc['NEact60h'] + dict_INRAPorc['kBR'] * FHP60) * (OnePig.LW ** 0.6)

    if PDAL<=PDFirstLimit:
        OnePig.FirstAALimit='none'
    px1AL = (NE_intake - NEmAL) / NEmAL + 1
    py1AL = PDAL * dict_INRAPorc['GEProtJaap']
    mr = (dict_INRAPorc['mrPV1'] * OnePig.PVPDmax - dict_INRAPorc['mrPV2'] * dict_INRAPorc['PVmr1'] - OnePig.LW * dict_INRAPorc['mrPV1'] + OnePig.LW * dict_INRAPorc['mrPV2']) / (
                     -dict_INRAPorc['PVmr1'] + OnePig.PVPDmax) * (NEmAL / 1000)
    F = 1 / 2 * (mr * (px1AL * px1AL) - 2 * px1AL * py1AL - mr) / (-py1AL + px1AL * mr - mr)
    if F <= 1 or (py1AL / (px1AL - 1)) <= mr:  # Minimum
        F = float('inf')
        a = py1AL / (px1AL - 1)
        b = 0
    else:
        if mr >= 0:
            a = -(px1AL * mr - mr - 2 * py1AL) / (px1AL - 1)
            b = (-py1AL + px1AL * mr - mr) / (px1AL * px1AL - 2 * px1AL + 1)
        else:
            a = 2 * py1AL / (F - 1)
            b = -py1AL / ((F - 1) * (F - 1))

    if px1AL < F:
        PDmaxE = a * (F - 1) + b * (F - 1) * (F - 1)
    else:
        PDmaxE = py1AL
    px1 = (NE_intake - NEmAL) / NEmAL + 1

    if px1 < F:
        py1 = a * (px1 - 1) + b * (px1 - 1) * (px1 - 1)
    else:
        py1 = PDmaxE
    PD = min(py1 / dict_INRAPorc['GEProtJaap'], PDFirstLimit)
    if (py1 / dict_INRAPorc['GEProtJaap']) < PDFirstLimit:
        OnePig.FirstAALimit = 'NE'    
    
    # Apport en énergie
    PfreeNE = 0
    TabProtFreeME = np.zeros(6)
    TabProtFreeNE = np.zeros(6)
    for i in range(6):
        TabProtFreeME[i] = TabDEIntake[i] * dict_INRAPorc['ValEnergy'][i][3] * dict_INRAPorc['ValEnergy'][i][1]
        TabProtFreeNE[i] = TabProtFreeME[i] * dict_INRAPorc['ValEnergy'][i][4]
        PfreeNE = PfreeNE + TabProtFreeNE[i]

    # Excess of protein
    ExcessProt = TabDEIntake[0] - PD  # Excess of protein, not retained in body gain
    ObligUrinELoss = 168 * (OnePig.LW ** 0.175)
    UrinEnergy = (ExcessProt / 6.25) * dict_INRAPorc['VarUrinELoss'] + ObligUrinELoss  # Energie urinaire
    MEexcessProt = ExcessProt * dict_INRAPorc['GEProtJaap'] - UrinEnergy  # Energie métabolisable perdue dans l'urine    
    NEexcessProt = MEexcessProt * dict_INRAPorc['kProtJaap']  # Energie nette perdue dans l'urine
    PDfreeNEintake = PfreeNE + NEexcessProt

    
    # Besoin en énergie pour le dépot de protéines
    PDfreeNEPD = PD * dict_INRAPorc['GEProtJaap'] * dict_INRAPorc['NEPD']
    PDfreeNEreq = PDfreeNEPD + NEm

    # Dépôt de lipides
    EnergyLD = PDfreeNEintake - PDfreeNEreq
   
    LD = EnergyLD / dict_INRAPorc['ValEnergy'][1][1]

    # Performances
    Protold = OnePig.Prot
    Lipold = OnePig.Lip
    OnePig.Prot = Protold + PD / 1000
    OnePig.Lip = Lipold + LD / 1000
    EBW = OnePig.CalcEBW(OnePig.LW)
    if OnePig.Prot > 0 and OnePig.Lip > 0:  # Calcul du poids vif vide (PVV) à partir des poids de protéines et lipides
        EBWafter = (dict_INRAPorc['Pallom'] * ((OnePig.Prot * 1000) ** dict_INRAPorc['Ballom']) + dict_INRAPorc['Lallom'] * (
                    (OnePig.Lip * 1000) ** dict_INRAPorc['Ballom'])) / 1000
    else:
        EBWafter = EBW
    BWafter = OnePig.CalcBW(EBWafter)

    # excretion
    OnePig.OMexc += FM_intake * (((Feed1.OM/1000) * Pctfeed1/100 + (Feed2.OM/1000) * (1-Pctfeed1/100)) - ((Feed1.dOM/1000) * Pctfeed1/100 + (Feed2.dOM/1000) * (1-Pctfeed1/100)))
    OnePig.dFibre += FM_intake * (Feed1.dFibre * Pctfeed1/100 + Feed2.dFibre * (1-Pctfeed1/100))

    
    # Nitrogen
    IngN = FM_intake * ((Feed1.CP/1000) * Pctfeed1/100 + (Feed2.CP/1000) * (1-Pctfeed1/100)) * 0.16  # N_intake
    DigN = TabDEIntake[0] * 0.16/1000  # digested N
    PDN = (PD/1000) * 0.16  # potential N retained in body mass
    OnePig.FecalN = IngN - DigN  # Excretion of fecal nitrogen
    OnePig.UrinN = max(DigN - PDN, 0)  # urinary nitrogen
    OnePig.RetainedN = min(DigN, PDN)  # N retained in body mass
    OnePig.N_intake += IngN # en kg
    OnePig.N_excreted += OnePig.FecalN + OnePig.UrinN  # En kg
    OnePig.TAN += OnePig.UrinN
    OnePig.DG = BWafter - OnePig.LW
    OnePig.LW = BWafter
    OnePig.Cumulated_FM_intake += FM_intake
    
    return OnePig