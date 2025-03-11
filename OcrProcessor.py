from dataclasses import dataclass, field
from enum import StrEnum, Enum
import subprocess
from typing import Protocol
import re

import image_processing

type BoundingBox = tuple[int,int,int,int]

class TesseractLanguage(StrEnum):
    FRENCH = "fra"

# information regarding Tesseract page segmentation modes can be found here:https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
class TesseractPsm(Enum):
    DEFAULT = 3
    SINGLE_WORD = 8
    SINGLE_TEXT_LINE = 7
    SEVERAL_TEXT_LINES = 6 # single font face without any variation
    SINGLE_CHARACTER = 10

@dataclass
class TesseractOcr:
    language: TesseractLanguage
    text_image_path: str
    _page_segmentation_mode: TesseractPsm = field(default_factory= lambda: TesseractPsm.DEFAULT)

    # oem 3, default parameter, see tesseract --help-oem in cmd for more details
    # more info here: https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc 
    def run_ocr(self):
        return subprocess.getoutput(f'tesseract {self.text_image_path} - -l {self.language.value} --oem 3 --psm {self._page_segmentation_mode.value}')
    
    def set_page_segmentation_mode(self, psm:TesseractPsm) -> None:
        self._page_segmentation_mode = psm

@dataclass
class OrcProcessor:
    '''
    Performs OCR on the text extracted from the table.
    - image: the mathematical representation of the image containing the table previously extracted.
    - original_image_path: the path to the original image, used to apply naming conventions to the image slices created.
    - images_folder_path: the path to the folder where the cropped text image are stored
    - table_column_names: a list containing the name of the columns from the table, from left to right. Size should be the same as the expected number of columns.
    - language: the language of the text we want to recognize with OCR
    '''
    table_bounding_box_array: list
    image: image_processing.Image
    original_image_path: str
    images_folder_path: str
    table_column_names: list[str]
    language: TesseractLanguage

    def run(self) -> list[list[str]]:
        table = []
        image_number = 0
        for i in range(len(self.table_bounding_box_array)):
            text_current_row = []
            # get the rows bounding boxes per columns, for the i-th row of the table
            column_rows = [self.table_bounding_box_array[i][0][j] for j in range(len(self.table_column_names))]
            # creates the slices for the OCR for the rows bounding boxes in the columns
            for k in range(len(column_rows)):
                cropped_text_boxes_images = self.slice_text_box_from_image(column_row_bounding_boxes=column_rows[k])
                ocr_outputs = []
                # gets the text recognized at the column level
                for box in cropped_text_boxes_images:
                    box_path = self.store_cropped_image(col_name=self.table_column_names[k], image_number=image_number, image=box)
                    ocr_engine = TesseractOcr(language=self.language, text_image_path=box_path)
                    output = self.apply_ocr(ocr_engine=ocr_engine)
                    ocr_outputs.append(output)
                    image_number += 1
                text_current_row.append(" ".join(ocr_outputs))
            table.append(text_current_row)
        return table  

    def slice_text_box_from_image(self, column_row_bounding_boxes:list[BoundingBox]) -> list[image_processing.Image]:
        '''
        Returns a list of the image coordinates corresponding to the text detected in this column's row
        - column_row_bounding_boxes: list containing all the bounding boxes which belong to this column's row.
        Boxes are in the format (top-left-corner_x, top-left-corner_y, box_width, box_height)
        '''
        return [
            image_processing.crop_image(
                image=self.image, 
                height_boundaries=(box[1], box[1] + box[3]), 
                width_boundaries=(box[0], box[0] + box[2])
                ) for box in column_row_bounding_boxes
            ]
        
    def store_cropped_image(self, col_name:int, image_number:int, image:image_processing.Image) -> str:
        '''
        - col_name: the name of the column in the table to extract
        - image_number: the order number at which the image was processed
        - image: the mathematical representation of the cropped image to store
        '''
        cropped_image_name = col_name + str(image_number) + ".jpg"
        img_handler = image_processing.ImageHandler(image_path=self.original_image_path)
        path = img_handler.store_image(folder_path=self.images_folder_path, file_name=cropped_image_name, image=image)
        return path
    
    def clean_ocr_output(self, output:str):
        '''
        Removes '\n' char for the .csv creation, and removes any leading or ending whitespace.
        - output: the text extracted via OCR
        '''
        new_line_pattern = r'\r?\n'
        if len(re.findall(new_line_pattern, output)) > 0:
            output = re.sub(new_line_pattern, ' ', output)
        return output.strip()
    
    def apply_ocr(self, ocr_engine:TesseractOcr) -> str:
        '''
        Applies ocr for the slice provided.
        - ocr_engine: the tesseract OCR engine, configured for the slice.
        '''
        # most of the time, we will perform OCR on single lines of text
        ocr_engine.set_page_segmentation_mode(psm=TesseractPsm.SINGLE_TEXT_LINE)
        output = ocr_engine.run_ocr()
        # if OCR fails, most presumably we provided more than a line, so we need to apply OCR with a different mode
        if output == "":
            ocr_engine.set_page_segmentation_mode(psm=TesseractPsm.SEVERAL_TEXT_LINES)
            output = ocr_engine.run_ocr()
        return self.clean_ocr_output(output=output)


def main():
    images_folder_path = "images/ocr_slices/"
    img_handler = image_processing.ImageHandler(image_path="images/debug/IMG_0148_otsu.jpg")
    image = img_handler.load_image()
    table_array = [[[[(162, 202, 531, 52)], [(1139, 191, 578, 54)], [(1981, 169, 455, 73)]]], [[[(165, 363, 525, 71)], [(1140, 360, 312, 54), (1141, 441, 536, 55)], [(1984, 342, 404, 71), (1984, 425, 596, 72)]]], [[[(164, 610, 505, 71)], [(1142, 610, 510, 55)], [(1981, 597, 820, 70)]]], [[[(165, 777, 366, 54)], [(1143, 779, 576, 71)], [(1984, 767, 759, 75)]]], [[[(165, 944, 478, 70)], [(1146, 947, 654, 72)], [(1984, 940, 599, 72)]]], [[[(167, 1110, 501, 70), (168, 1190, 287, 54)], [(1147, 1115, 653, 55)], [(1983, 1111, 467, 62), (1981, 1191, 720, 73)]]], [[[(170, 1356, 655, 56)], [(1149, 1361, 548, 55)], [(1983, 1362, 679, 72)]]], [[[(170, 1522, 522, 70)], [(1150, 1526, 618, 56)], [(1981, 1530, 564, 65)]]], [[[(172, 1687, 579, 71)], [(1151, 1692, 434, 56), (1151, 1772, 359, 70)], [(1984, 1699, 723, 75), (1983, 1780, 504, 74), (1984, 1861, 797, 203)]]], [[[(169, 2149, 392, 55)], [(1149, 2154, 488, 59)], [(1983, 2167, 811, 79)]]], [[[(164, 2316, 437, 72)], [(1148, 2323, 428, 73), (1148, 2404, 437, 59)], [(1990, 2339, 638, 71), (1991, 2421, 523, 62)]]], [[[(164, 2565, 310, 56)], [(1147, 2573, 718, 83)], [(1989, 2592, 770, 78)]]], [[[(162, 2734, 429, 58)], [(1147, 2744, 333, 59), (1147, 2826, 365, 74)], [(1993, 2765, 190, 57), (1994, 2847, 789, 72), (1990, 2925, 839, 71)]]]]

    ocr_processor = OrcProcessor(
        table_bounding_box_array=table_array,
        image=image,
        original_image_path="images/IMG_0148.jpg",
        images_folder_path=images_folder_path,
        table_column_names=["meal_name", "wine_types", "wine_appelations"],
        language=TesseractLanguage.FRENCH
    )

    ocr_table = ocr_processor.run()
    print(ocr_table)
  

if __name__ == "__main__":
    main()