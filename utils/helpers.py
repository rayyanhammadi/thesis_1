import pandas as pd


class Helpers():

    def get_var_FullName(var_name:str, df) -> str:
        """
        A partir d'une nom de colonne, renvoie le nom complet de la variable correspondante
        - var_name : str, le nom de la colonne dans le dataframe BDD_data
        - return : str, le nom complet de la variable correspondante
        """
        if var_name not in df.columns:
            raise ValueError("{} not found in the database".format(var_name))
        else:
            return df.columns[list(df.columns).index(var_name) + 1]

# Test
# print(get_var_FullName("FalseVariable"))  # OK
print(get_var_FullName("gold"))