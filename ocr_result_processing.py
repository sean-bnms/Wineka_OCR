import re
from pathlib import Path

import pandas as pd


def clean_bullet_points(raw_table:list[list[str]]):
    # bullet points are wrongly recognized as these characters
    pattern = r"\.\s|\*\s|\+\s|\-\s"

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
    path = "outputs/" + csv_name + ".csv"
    df.to_csv(path_or_buf=Path(path), sep="|")
    return path


def main():
    ocr_results = [
        ['Avocats aux crevettes', 'Blancs fins et savoureux', 'Mâcon, beaujolais'], 
        ['Beignets de crevettes', '+ Blancs vifs . Blancs effervescents', '- Alsace riesling + Champagne, crémants'], 
        ['Beignets de légumes', 'Blancs frais et fruités', 'Côtes-de-provence, vin-de-corse'], 
        ['Bricks au thon', 'Rosés épicés et charnus', 'Languedoc, côtes-de-provence'], 
        ['Carpaccio de bœuf', 'Rouges tendres et fringants', 'Beaujolais, anjou-gamay'], 
        ['Carpaccio et tartare de saumon', 'Blancs séveux au fruité frais', 'Entre-deux-mers, picpoul-de-pinet (languedoc)'], 
        ['Cassolette de fruits de mer', 'Blancs vifs et minéraux', 'Pouilly-fumé, alsace riesling'], 
        ['Cassolette de poisson', 'Blancs fruités et onctueux', 'Saint-véran, minervois'], 
        ['Escargots au beurre d’ail', '- Blancs bien secs + Rosés épices', '+ Chablis bien minéral, quincy, reuilly, pouilly-fumé + Languedoc, costières-de-nîmes (tavel, lirac, côtes-de- provence, bandol, côtes-du-roussillon..)'], 
        ['Flammekueche', 'Blancs vifs et fruités', 'Alsace sylvaner, alsace pinot blanc'], ['Melon au jambon', '+ Rosés généreux + Blancs moelleux', '+ Tavel, lirac, bandol, bellet + Muscat-de-mireval'], 
        ['Pissaladière', 'Blancs tendres et aromatiques', 'Côtes-de-provence, patrimonio'], ['Pizza aux anchois', '+ Blancs secs + Rosés épicés', '+ Lirac * Lirac, tavel, côtes-de-provence, cassis, coteaux-varois-en-provencé']]
    
    table_column_names = ["Plat", "Type de vin", "Appelation"]
    
    clean_table = clean_bullet_points(raw_table=ocr_results) 
    store_table_as_csv(table=clean_table, column_names=table_column_names, csv_name="test")




if __name__ == "__main__":
    main()