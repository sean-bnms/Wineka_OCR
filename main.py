import TableExtractor as te
import TableLinesAndIconsRemover as tlir
import OcrToTableTool as ottt
import numpy as np
import pandas as pd
from pathlib import Path
import re

PINK_YELLOW_ORANGE_COLOR_LIMITS_DICT = {
    "lower" : np.array([0, 20,20]),
    "upper" : np.array([35,250,250])
}

PINK_RED_COLOR_LIMITS_DICT = {
    "lower" : np.array([150,20,20]),
    "upper" : np.array([180,250,250])
}

COLUMN_NAMES=['Plats', 'Type de vins', 'Appellation'] 
OCR_FINAL_PATH = "outputs_ocr"

def run_ocr(path_image: str):
    path_to_image = Path(path_image)
    # extracts table from its background
    table_extractor = te.TableExtractor(image_path=path_to_image, icon_dict_1=PINK_YELLOW_ORANGE_COLOR_LIMITS_DICT, icon_dict_2=PINK_RED_COLOR_LIMITS_DICT)
    extracted_table = table_extractor.execute()
    # removes lines and icons
    lines_remover = tlir.TableLinesAndIconsRemover(image=extracted_table, image_path=path_to_image, icon_dict_1=PINK_YELLOW_ORANGE_COLOR_LIMITS_DICT, icon_dict_2=PINK_RED_COLOR_LIMITS_DICT)
    image_without_lines = lines_remover.execute()
    # apply Tesseract OCR
    ocr_tool = ottt.OcrToTableTool(image=image_without_lines, original_image=extracted_table, image_path=path_to_image)
    ocr_tool.execute()

def process_ocr_df(df):
    new_rows = []
    unchanged_rows = []
    delimiter_pattern = r'(?<!\.)[+\*\.](?!\.)' #we remove as delimiter '...' which sometimes occur
    other_start_delimiter_pattern = r'^- '
    for index, row in df.iterrows():
        # processing
        wine_type_value = row['Type de vins']
        if len(re.findall(other_start_delimiter_pattern, wine_type_value)) > 0: #we need to remove - char at the beginning to not mess with appelations with hyphens
              wine_type_value = re.sub(other_start_delimiter_pattern, '+', wine_type_value)
        appellation_value = row['Appellation']
        if len(re.findall(other_start_delimiter_pattern, appellation_value)) > 0: #we need to remove - char at the beginning to not mess with appelations with hyphens
              appellation_value = re.sub(other_start_delimiter_pattern, '+', appellation_value)
        # Split the string by delimiters and filter out empty strings
        segments_wine_type = [segment.strip() for segment in re.split(delimiter_pattern, wine_type_value) if segment.strip()]
        segments_appellation = [segment.strip() for segment in re.split(delimiter_pattern, appellation_value) if segment.strip()]
        # If there are segments, create new rows; otherwise, keep the row unchanged
        if len(segments_wine_type) > 1 and len(segments_wine_type) == len(segments_appellation):
            for segment_w, segment_a in list(zip(segments_wine_type, segments_appellation)):
                new_row = row.copy()
                new_row['Type de vins'] = segment_w
                new_row['Appellation'] = segment_a
                new_rows.append(new_row)
        else:
            unchanged_rows.append(row)
    new_rows_df = pd.DataFrame(new_rows)
    unchanged_rows_df = pd.DataFrame(unchanged_rows)
    final_df = pd.concat([unchanged_rows_df, new_rows_df], ignore_index=True)
    return final_df

def load_df(path_ocr):
    return pd.read_csv(filepath_or_buffer=Path(path_ocr), delimiter="|", names=COLUMN_NAMES)

def generate_csv_file(df, path_image):
        path = Path("outputs/" + Path(path_image).stem + ".csv")
        df.to_csv(path_or_buf=path, sep="|")


def extract_text_as_csv(path_image):
    run_ocr(path_image=path_image)
    path_ocr = OCR_FINAL_PATH + "/" + Path(path_image).stem + ".csv"
    df = load_df(path_ocr=path_ocr)
    processed_df = process_ocr_df(df=df)
    generate_csv_file(df=processed_df, path_image=path_image)


# path_image = "images/IMG_0148.jpg"
# extract_text_as_csv(path_image=path_image)

