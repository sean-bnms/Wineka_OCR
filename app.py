import streamlit as st
import pandas as pd
from main import extract_text_as_csv
from pathlib import Path
import os
from PIL import Image

OCR_FINAL_PATH = "outputs"
TABLE_INPUT_PATH = "images"
DEBUG_PATH = "images/debug"


# files handling
def collect_images():
    img_path = []
    for file in os.listdir(Path("images")):
        if file.endswith(".jpg"):
            img_path.append(file)
    return img_path

def get_ocr(path_image):
    ocrNotPerformed = True
    for file in os.listdir(Path(OCR_FINAL_PATH)):
        path_file = Path(file)
        if path_file.stem == Path(path_image).stem:
            print("OCR already performed")
            ocrNotPerformed = False
            break
    if ocrNotPerformed:
        extract_text_as_csv(path_image=path_image)

# buttons next and previous
def next(): 
    st.session_state.counter += 1
def prev(): 
    st.session_state.counter -= 1

def load_df(path_ocr):
    return pd.read_csv(filepath_or_buffer=Path(path_ocr), delimiter="|")

# INITIALISATION
if 'counter' not in st.session_state: 
    st.session_state.counter = 0

table_imgs = collect_images()

st.set_page_config(layout="wide")

def st_main():
    container = st.empty()
    cols = st.columns(2)
    with cols[1]: st.button("Next ➡️", on_click=next, use_container_width=True)
    with cols[0]: st.button("⬅️ Previous", on_click=prev, use_container_width=True)

    with container.container():
        tab1, tab2 = st.tabs(["OCR Results", "Debug OCR"])

        with tab1:
            ## Select image based on the current counter
            path_image = TABLE_INPUT_PATH + "/" + table_imgs[st.session_state.counter]
            image = Image.open(Path(path_image)).rotate(-90)

            col1, col2 = st.columns(2)
            with col1:
                st.header(f"Input image: {Path(path_image).stem} ")
                st.image(image=image)
            get_ocr(path_image=path_image)
            path_ocr = OCR_FINAL_PATH + "/" + Path(path_image).stem + ".csv"
            processed_df = load_df(path_ocr=path_ocr)   
            with col2:
                with st.form("my_form"):
                    st.header("OCR Table")
                    edited_df = st.data_editor(processed_df, num_rows="dynamic", hide_index=True)
                    submitted = st.form_submit_button("Submit")
                if submitted:
                    print("Edited Dataframe")
                    edited_df.to_csv(path_or_buf=Path(path_ocr), sep="|")
        
        with tab2:
            path_image_table_extraction = DEBUG_PATH + "/table_extraction/" + table_imgs[st.session_state.counter].split(".jpg")[0] + "_8_perspective_corrected_with_padding.jpg"
            image_table_extraction = Image.open(Path(path_image_table_extraction))
            path_image_lines_removal = DEBUG_PATH + "/lines_removal/" + table_imgs[st.session_state.counter].split(".jpg")[0] + "_10_image_without_lines_noise_removed.jpg"
            image_lines_removal = Image.open(Path(path_image_lines_removal))
            path_image_ocr = DEBUG_PATH + "/ocr/" + table_imgs[st.session_state.counter].split(".jpg")[0] + "_4_updated_bounding_boxes.jpg"
            image_ocr = Image.open(Path(path_image_ocr))
            col1, col2, col3 = st.columns(3)

            with col1:
                st.header(f"Table extraction")
                st.image(image=image_table_extraction)
            with col2:
                st.header(f"Lines/icons removal")
                st.image(image=image_lines_removal)
            with col3:
                st.header(f"Text box detection")
                st.image(image=image_ocr)
            

st_main()

