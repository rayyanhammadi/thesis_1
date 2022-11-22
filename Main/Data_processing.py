import math
import numpy as np
import pandas as pd

class Data:
    def __init__(self,BDD_path, BDD_sheet):
        self.raw_df = pd.read_excel(BDD_path, sheet_name=BDD_sheet) #pour lire le fichier excel
        self.df = None

    def data_processing(self):
        self.df = self.raw_df.iloc[1:,:]   #on enleve la premiere ligne qui contient les longs titres des colonnes
        self.df.columns = self.raw_df.iloc[0,:] # Remplacement des titres des colonnes par la...
        self.df.set_index("dates", drop=True, inplace=True) # ...ligne des noms courts des colonnes

        self.df = self.df.astype("float") # Conversion en nombre à décimales (en utilisant float), sinon string (cad chaîne de texte) en format par défaut

        print("Data processed succesfully")
        return(self.df) #on return le tableau de données dont on a remplacé les titres des colonnes par la première ligne

    def data_summary(self):   #on print le summary du tableau de données
        print(self.df.head())
        print(self.df.describe())


