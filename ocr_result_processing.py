import re
from pathlib import Path
import pandas as pd


def clean_bullet_points(raw_table:list[list[str]]):
    # bullet points are wrongly recognized as these characters
    pattern = r"\.\s|\*\s|\+\s|-\s"

    clean_list = []
    for row in raw_table:
        wine_types = row[1]
        wine_appelations = row[2]
        # checks whether we have bullet points incorrectly identified on both 2 last columns
        if len(re.findall(pattern, wine_types)) == len(re.findall(pattern, wine_appelations)) and len(re.findall(pattern, wine_types)) > 0:
            # the first element of the re.split() is not text as inital text starts with a bullet point
            cleaned_wine_types = [wine_type.strip() for wine_type in re.split(pattern, wine_types)[1:]]
            cleaned_wine_appelations = [wine_type.strip() for wine_type in re.split(pattern, wine_appelations)[1:]]
            # creates a new line for each bullet point item
            for w_type, w_appelation in list(zip(cleaned_wine_types, cleaned_wine_appelations)):
                new_line = [row[0], w_type, w_appelation]
                clean_list.append(new_line)     
        else: 
            clean_list.append(row)
    return clean_list

def store_table_as_csv(table:list[list[str]], column_names:list[str], csv_name:str, row_delimiter: str = "|") -> str:
    '''
    Stores the table as a .csv file.
    - table: table to store in the csv
    - column_names: name of the columns of the csv
    - csv_name: the name of the csv file, without the extension
    - row_delimiter: delimiter for the file, defaulted to '|'
    '''
    df = pd.DataFrame(data=table, columns=column_names)
    path = Path("outputs/" + csv_name + ".csv")
    df.to_csv(path_or_buf=path, sep="|")


def main():
    ocr_results = [
        ['Andouille de Guéméné', 'Blancs vifs et secs', 'Vouvray, muscadet, Sancerre, quincy, reuilly'], 
        ['Andouillette grillée', '. Blancs vifs + Rouges souples', '. Chablis, saint-bris, mâcon - Mâcon, beaujolais-villages, Minervois'], 
        ['Boudin blanc', '. Blancs vifs et minéraux . Effervescents', '. Alsace riesling, alsace pinot blanc, savennières, pouilly-fumé + Champagne'], 
        ['Boudin noir', 'Rouges charnus de caractère', 'Corbières, saint-joseph, cornas, saint-émilion, madiran, cahors, béarn'], 
        ['Fromage de tête', '. Blancs vifs et savoureux + Rouges fruités et vifs', '- Beaujolais, petit-chablis, saint-bris + Chiroubles, coteaux-du-quercy'], 
        ['Jambon cru', '+ Blancs savoureux + Rouges charnus', '+ Rancio sec du Roussillon (IGP), collioure + Gigondas, irouléguy, patrimonio, madiran'], 
        ['Jambon cuit et persillé', '+ Blancs tendres . Rouges souples', '+ Graves, marsannay + Saint-nicolas-de-bourgueil, irancy, bou rgogne-passetoutgrain, beaujolais-villages'], 
        ['Pâte de campagne', 'Rouges fruités et fringants', 'Crus du Beaujolais (saint-amour, fleurie, brouilly, côte-de-brouilly, juliénas...) et tous les rouges gouleyants à base de gamay où de grenache'], 
        ['Rillettes d’oie', 'Blancs expressifs et vifs', 'Bergerac, alsace riesling, sance reuilly, quincy, anjou']
        ]
    
    table_column_names = ["Plat", "Type de vin", "Appelation"]
    
    clean_table = clean_bullet_points(raw_table=ocr_results) 
    store_table_as_csv(table=clean_table, column_names=table_column_names, csv_name="test")



if __name__ == "__main__":
    main()