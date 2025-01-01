import cv2
import numpy as np
import subprocess
from pathlib import Path
import re

class OcrToTableTool:

    def __init__(self, image, original_image, image_path):
        self.thresholded_image = image
        self.original_image = original_image
        self.image_path = image_path
        self.new_line_pattern = r'\r?\n'
    
    def execute(self):
        self.dilate_image()
        self.store_process_image('0_dilated_image.jpg', self.dilated_image)
        self.find_contours()
        self.store_process_image('1_contours.jpg', self.image_with_contours_drawn)
        self.convert_all_contours_to_bounding_boxes()
        self.store_process_image('2_bounding_boxes.jpg', self.image_with_all_bounding_boxes)
        self.convert_incorrect_contours_to_bounding_boxes()
        self.store_process_image('3_incorrect_bounding_boxes.jpg', self.image_with_incorrect_bounding_boxes)
        self.clean_bounding_box_list()
        self.visualize_updated_bounding_boxes()
        self.store_process_image('3_updated_bounding_boxes.jpg', self.image_with_updated_bounding_boxes)
        self.sort_bounding_boxes_by_x_coordinate()
        self.sort_ordered_bounding_boxes_by_columns()
        self.convert_updated_contours_to_bounding_boxes()
        self.store_process_image('4_updated_incorrect_bounding_boxes.jpg', self.image_with_updated_incorrect_bounding_boxes)
        self.clean_bounding_box_list()
        self.visualize_updated_bounding_boxes()
        self.store_process_image('4_updated_bounding_boxes.jpg', self.image_with_updated_bounding_boxes)
        self.collect_ordered_columns()
        self.collect_ordered_rows()
        self.get_array()
        self.crop_each_bounding_box_and_ocr()
        self.generate_csv_file()
    
    def dilate_image(self):
        kernel_to_remove_gaps_between_words = np.array([
                [1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,1],
        ])
        self.dilated_image = cv2.dilate(self.thresholded_image, kernel_to_remove_gaps_between_words, iterations=5)
        simple_kernel = np.ones((5,5), np.uint8)
        self.dilated_image = cv2.dilate(self.dilated_image, simple_kernel, iterations=2)
    
    def find_contours(self):
        result = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        # The code below is for visualization purposes only.
        # It is not necessary for the OCR to work.
        self.image_with_contours_drawn = self.original_image.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 3)
    
    def convert_all_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.bounding_boxes.append((x, y, w, h))
            # This line below is about
            # drawing a rectangle on the image with the shape of
            # the bounding box. Its not needed for the OCR.
            # Its just added for debugging purposes.
            self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)

    def get_mean_height_of_bounding_boxes(self):
        heights = []
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)
        return np.mean(heights)
    
    # to account for small portions of table lines who were not completely eroded and other small unwanted noises
    def convert_incorrect_contours_to_bounding_boxes(self):
        self.incorrect_boxes = []
        mean_box_height = self.get_mean_height_of_bounding_boxes()
        self.image_with_incorrect_bounding_boxes = self.original_image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < mean_box_height / 1.5: #words box have more or less the same size, under this threshold it is certainly a line
                self.incorrect_boxes.append((x, y, w, h))
                self.image_with_incorrect_bounding_boxes = cv2.rectangle(self.image_with_incorrect_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)
    
    def clean_bounding_box_list(self):
        for bounding_box in self.bounding_boxes.copy():
            if bounding_box in self.incorrect_boxes:
                self.bounding_boxes.remove(bounding_box)
    
    def visualize_updated_bounding_boxes(self):
        self.image_with_updated_bounding_boxes = self.original_image.copy()
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            self.image_with_updated_bounding_boxes = cv2.rectangle(self.image_with_updated_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)
    
    def sort_bounding_boxes_by_x_coordinate(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[0])
    
    def sort_ordered_bounding_boxes_by_columns(self):
        self.columns = {}
        k = 1
        self.columns[f"{k}"] = [self.bounding_boxes[0]]
        mid_interval = self.columns[f"{k}"][0][0]
        column_x_values = [mid_interval]
        for i in range(1,len(self.bounding_boxes)):
            x = self.bounding_boxes[i][0]
            column_start_interval = range(mid_interval - 30, mid_interval + 30)
            if x in column_start_interval:
                self.columns[f"{k}"].append(self.bounding_boxes[i])
                column_x_values.append(self.bounding_boxes[i][0])
                #we want to have a centered mid_interval, as we have a tendency to see x increase for a column because of the perspective distortion
                mid_interval = sum(column_x_values) // len(column_x_values) 
            else:
                k+=1
                self.columns[f"{k}"] = [self.bounding_boxes[i]]
                column_x_values = [self.bounding_boxes[i][0]]
                mid_interval = self.columns[f"{k}"][0][0]
    
    # to account for bigger portions of table lines who were not completely eroded, often in the middle of two columns of the table
    def convert_updated_contours_to_bounding_boxes(self):
        self.image_with_updated_incorrect_bounding_boxes = self.original_image.copy()
        columns_length = {k : len(self.columns[k]) for k in self.columns.keys()}
        if len(columns_length) > 3: #we expect only 3 columns for correct images
            while len(columns_length) != 3:
                min_key = min(columns_length, key = columns_length.get)
                self.incorrect_boxes += self.columns[min_key]
                del columns_length[min_key]
                del self.columns[min_key]
        for box in self.incorrect_boxes:
            x, y, w, h = box
            self.image_with_updated_incorrect_bounding_boxes = cv2.rectangle(self.image_with_updated_incorrect_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)
        
    
    def collect_ordered_columns(self):
        columns_x = {k : self.columns[k][0][0] for k in self.columns.keys()}
        keys = [k for k in self.columns.keys()]
        min_key = min(columns_x, key = columns_x.get)
        max_key = max(columns_x, key = columns_x.get)
        keys.remove(min_key)
        keys.remove(max_key)
        mid_key = keys[0] #one key left 
        first_column = self.columns[min_key]
        second_column = self.columns[mid_key]
        third_column = self.columns[max_key]
        self.ordered_columns = {
            "1": sorted(first_column, key=lambda x: x[1]),
            "2": sorted(second_column, key=lambda x: x[1]),
            "3": sorted(third_column, key=lambda x: x[1])
        }
        print("Final Columns: ", self.ordered_columns)
    

    # def collect_ordered_rows(self):
    #     mean_box_height = self.get_mean_height_of_bounding_boxes()
    #     self.ordered_rows = {}
    #     k = 1
    #     # x and y are the top left coordinates of the box, (x + w), (y + h) are the bottom right ones
    #     # apply a distance to discriminate if two consecutive boxes in a column are from the same row or not
    #     for column in self.ordered_columns.keys():
    #         column_boxes = self.ordered_columns[column]
    #         self.ordered_rows[column] = {}
    #         self.ordered_rows[column][f"{k}"] = [column_boxes[0]]
    #         if len(column_boxes) == 1: #1 box only, in one row only
    #             pass
    #         else:
    #             for i in range(0,len(column_boxes)-1):
    #                 x1, y1, w1, h1 = column_boxes[i]
    #                 barycenter_y1 = y1 + h1//2
    #                 x2, y2, w2, h2 = column_boxes[i+1]
    #                 barycenter_y2 = y2 + h2//2
    #                 if abs(barycenter_y2 - barycenter_y1) < 1.5 * mean_box_height: #consecutive boxes, same row
    #                     self.ordered_rows[column][f"{k}"].append(column_boxes[i+1])
    #                 else:
    #                     k+=1
    #                     self.ordered_rows[column][f"{k}"] = [column_boxes[i+1]]
    #             k = 1
    #     print("\n Rows per columns: ", self.ordered_rows)

    def collect_ordered_rows(self):
        mean_box_height = self.get_mean_height_of_bounding_boxes()
        self.ordered_rows = {}
        k = 1
        # x and y are the top left coordinates of the box, (x + w), (y + h) are the bottom right ones
        # apply a distance to discriminate if two consecutive boxes in a column are from the same row or not
        for column in self.ordered_columns.keys():
            column_boxes = self.ordered_columns[column]
            self.ordered_rows[column] = {}
            self.ordered_rows[column][f"{k}"] = [column_boxes[0]]
            if len(column_boxes) == 1: #1 box only, in one row only
                pass
            else:
                for i in range(0,len(column_boxes)-1):
                    x1, y1, w1, h1 = column_boxes[i]
                    bottom_1 = y1 + h1
                    x2, y2, w2, h2 = column_boxes[i+1]
                    top_2 = y2 
                    if abs(top_2 - bottom_1) < mean_box_height // 2: #consecutive boxes, same row
                        self.ordered_rows[column][f"{k}"].append(column_boxes[i+1])
                    else:
                        k+=1
                        self.ordered_rows[column][f"{k}"] = [column_boxes[i+1]]
                k = 1
        print("\n Rows per columns: ", self.ordered_rows)
    
    
    def get_array(self):
        self.array = []
        rows = self.ordered_rows
        common_row_keys = list(rows['1'].keys() & rows['2'].keys() & rows['3'].keys())
        row_numbers = [int(key) for key in common_row_keys]
        ordered_row_numbers = sorted(row_numbers)
        for i in range(len(ordered_row_numbers)):
            row = []
            key = str(ordered_row_numbers[i])
            triplet = [rows['1'][key], rows['2'][key], rows['3'][key]]
            row.append(triplet)
            self.array.append(row)
        print("\n Final array: ", self.array)


    def crop_each_bounding_box_and_ocr(self):
        self.table = []
        current_row = []
        image_number = 0
        for i in range(len(self.array)):
            col1 = self.array[i][0][0]
            col2 = self.array[i][0][1]
            col3 = self.array[i][0][2]
            current_row_col1, image_number = self.create_row_colum_slice(col=col1, col_label="col1", image_number=image_number)
            current_row.append(" ".join(current_row_col1))
            current_row_col2, image_number = self.create_row_colum_slice(col=col2, col_label="col2", image_number=image_number)
            current_row.append(" ".join(current_row_col2))
            current_row_col3, image_number = self.create_row_colum_slice(col=col3, col_label="col3", image_number=image_number)
            current_row.append(" ".join(current_row_col3))
            self.table.append(current_row)
            current_row = []
        print("RESULT OCR: ", self.table)


    def create_row_colum_slice(self, col, col_label, image_number):
        image_number = image_number
        current_row_col = []
        for j in range(len(col)):
                x, y, w, h = col[j]
                cropped_image = self.original_image[y:y+h, x:x+w]
                image_slice_path = Path("images/ocr_slices/" + self.image_path.stem + "_" + col_label + "_" + str(image_number) + ".jpg")
                cv2.imwrite(str(image_slice_path.resolve()), cropped_image) #OpenCV cannot handle Path objects, it expects strings
                results_from_ocr = self.get_result_from_tesseract(str(image_slice_path.resolve()))
                current_row_col.append(results_from_ocr)
                image_number += 1
        return current_row_col, image_number
    
    
    def get_result_from_tesseract(self, image_path):
        # l fra for French language
        # psm 7 as we input single lines of text, more info here: https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
        # if more than a line, we use psm 6
        # oem 3, default parameter, see tesseract --help-oem in cmd for more details
        # more info here: https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc 
        output = subprocess.getoutput('tesseract ' + image_path + ' - -l fra --oem 3 --psm 7')
        if output == "":
            output = subprocess.getoutput('tesseract ' + image_path + ' - -l fra --oem 3 --psm 6')
            if len(re.findall(self.new_line_pattern, output)) > 0: #we need to remove '\n' char for the .csv creation
                output = re.sub(self.new_line_pattern, ' ', output)
        print("\n OUTPUT OCR: \n", output)
        output = output.strip()
        return output
    
    def generate_csv_file(self):
        path = Path("outputs_ocr/" + self.image_path.stem + ".csv")
        with open(path, "w", encoding='utf-8') as f:
            for row in self.table:
                f.write("|".join(row) + "\n")

    def store_process_image(self, file_name, image):
        path = Path("images/debug/ocr/" + self.image_path.stem + "_" + file_name)
        cv2.imwrite(str(path.resolve()), image) #OpenCV cannot handle Path objects, it expects strings


     
