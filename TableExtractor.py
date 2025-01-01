# Mainly inspired by: https://livefiredev.com/how-to-extract-table-from-image-in-python-opencv-ocr/ 
# Changes are adaptation to colored background and non-defined external contour for the table
# Storing of debug files was also adapted to our use case

import cv2
import numpy as np
from pathlib import Path


class TableExtractor:
    def __init__(self, image_path, icon_dict_1, icon_dict_2):
        self.image_path = image_path 
        self.color_limit_1 = icon_dict_1
        self.color_limit_2 = icon_dict_2
    
    def execute(self):
        self.read_image()
        #pre-processing to improve the contour detection
        print(f"Preprocessing START: ")
        self.filter_background_color()
        self.store_process_image("0_bckgd_filtered.jpg", self.background_color_filtered_image)
        self.convert_image_to_grayscale()
        self.store_process_image("1_grayscaled.jpg", self.grayscale_image)
        self.threshold_image()
        self.store_process_image("2_thresholded.jpg", self.thresholded_image)
        self.invert_image()
        self.store_process_image("3_inverteded.jpg", self.inverted_image)
        self.dilate_image()
        self.store_process_image("4_dilated.jpg", self.dilated_image)
        #isolates table from the background
        print(f"Table extraction START: ")
        self.find_contours()
        self.store_process_image("5_all_contours.jpg", self.image_with_all_contours)
        table_corner_edges = self.get_table_corner_edges()
        print(f"Edges are, from TOP LEFT to BOTTOM LEFT (clockwise), {table_corner_edges}")
        self.visualize_table_corner_edges(table_corner_edges=table_corner_edges)
        self.store_process_image("6_only_table_corner_edges.jpg", self.image_with_only_table_corner_edges)
        self.calculate_new_width_and_height_of_image(table_corner_edges=table_corner_edges)
        print(self.new_image_width, self.new_image_height)
        self.apply_perspective_transform(table_corner_edges=table_corner_edges)
        self.store_process_image("7_perspective_corrected.jpg", self.perspective_corrected_image)
        self.add_10_percent_padding()
        self.store_process_image("8_perspective_corrected_with_padding.jpg", self.perspective_corrected_image_with_padding)
        return self.perspective_corrected_image_with_padding

    def read_image(self):
        self.image = cv2.imread(str(self.image_path.resolve())) #OpenCV cannot handle Path objects, it expects strings


    def filter_background_color(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        #mask1
        lower1 = self.color_limit_1["lower"]
        upper1 = self.color_limit_1["upper"]
        mask_1 = cv2.inRange(hsv, lower1, upper1) 
        #mask2
        lower2 = self.color_limit_2["lower"]
        upper2 = self.color_limit_2["upper"]
        mask_2 = cv2.inRange(hsv, lower2, upper2) 
        mask = mask_1 + mask_2
        # apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # invert morp image
        mask = 255 - morph
        # apply mask to image
        self.background_color_filtered_image = cv2.bitwise_and(self.image, self.image, mask=mask)

    def convert_image_to_grayscale(self):
        self.grayscale_image = cv2.cvtColor(self.background_color_filtered_image, cv2.COLOR_BGR2GRAY)
    
    def threshold_image(self):
        self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #OTSU method finds the best threshold possible for binary color attribution

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)
    
    def dilate_image(self):
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=5)

    def find_contours(self):
        self.contours, self.hierarchy = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Below lines are added to show all contours, for debugging purposes
        self.image_with_all_contours = self.image.copy()
        cv2.drawContours(self.image_with_all_contours, self.contours, -1, (0, 255, 0), 3)

    def get_table_corner_edges(self):
        # Initiate variables to track the table angle coordinates
        height = self.image.shape[0]
        width =  self.image.shape[1]
        x_table_min, x_table_max, y_table_min, y_table_max = width, 0, height, 0
        coordinates = []
        table_corner_edges = []

        self.contours, self.hierarchy = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in self.contours:
            perimeter = cv2.arcLength(contour, True) #True means the contour is expected to be closed
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            # Used to flatted the array containing the co-ordinates of the vertices. 
            n = approx.ravel() 
            i = 0

            for j in n : 
                if(i % 2 == 0): 
                    x = n[i] 
                    y = n[i + 1] 
                    coordinates.append((x,y))
                    if  10 < x < width - 10: # image edges need to be removed, put a 10px margin
                        if x < x_table_min:
                            x_table_min = x
                        if x > x_table_max:
                            x_table_max = x
                    if  10 < y < height - 10: # image edges need to be removed, put a 10px margin
                        if y < y_table_min:
                            y_table_min = y
                        if y > y_table_max:
                            y_table_max = y
                i = i + 1

        # String containing the co-ordinates. 
        string_top_left = (x_table_min, y_table_min) 
        string_bottom_left = (x_table_min, y_table_max) 
        string_top_right = (x_table_max, y_table_min) 
        string_bottom_right = (x_table_max, y_table_max) 
        optimal_edges = [string_top_left, string_top_right, string_bottom_right, string_bottom_left]
        print(optimal_edges)

        for j in range(len(optimal_edges)):
            distances = [self.calculate_distance_between_points(optimal_edges[j][0], optimal_edges[j][1], coordinates[i][0], coordinates[i][1]) for i in range(len(coordinates))]
            x_real, y_real = coordinates[distances.index(min(distances))]
            table_corner_edges.append((x_real, y_real))
        return table_corner_edges
    
    def calculate_distance_between_points(self, x0,y0,x1,y1):
        return np.sqrt((x0-x1)**2 + (y0-y1)**2)
    
    def visualize_table_corner_edges(self, table_corner_edges):
        self.image_with_only_table_corner_edges = self.image.copy()
        for x,y in table_corner_edges:
            # Display coordinates 
            cv2.putText(self.image_with_only_table_corner_edges , str(x) + " " + str(y), (x, y), 
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0)) 
            cv2.circle(self.image_with_only_table_corner_edges , (x,y), 10, (255, 0, 0), -1)    
  
    def calculate_new_width_and_height_of_image(self, table_corner_edges):
        existing_image_width = self.image.shape[1]
        existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
        distance_between_top_left_and_top_right = self.calculate_distance_between_points(table_corner_edges[0][0], table_corner_edges[0][1], table_corner_edges[1][0], table_corner_edges[1][1])
        distance_between_top_left_and_bottom_left = self.calculate_distance_between_points(table_corner_edges[0][0], table_corner_edges[0][1], table_corner_edges[3][0], table_corner_edges[3][1])
        aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right
        self.new_image_width = existing_image_width_reduced_by_10_percent
        self.new_image_height = int(self.new_image_width * aspect_ratio)
    
    def apply_perspective_transform(self, table_corner_edges):
        pts1 = np.float32([ [table_corner_edges[i][0], table_corner_edges[i][1]] for i in range(len(table_corner_edges))])
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height], [0, self.new_image_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.perspective_corrected_image = cv2.warpPerspective(self.image, matrix, (self.new_image_width, self.new_image_height))
    
    def add_10_percent_padding(self):
        image_height = self.image.shape[0]
        padding = int(image_height * 0.1)
        self.perspective_corrected_image_with_padding = cv2.copyMakeBorder(self.perspective_corrected_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    def store_process_image(self, file_name, image):
        path = Path("images/debug/table_extraction/" + self.image_path.stem + "_" + file_name)
        cv2.imwrite(str(path.resolve()), image) #OpenCV cannot handle Path objects, it expects strings
    


