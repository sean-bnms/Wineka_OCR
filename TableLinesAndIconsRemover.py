# To understand the process of removing the table lines with dilation/erosion, https://docs.opencv.org/4.x/dd/dd7/tutorial_morph_lines_detection.html 
# To erode icons with HSV color space: https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv/48367205#48367205
import cv2
import numpy as np
from pathlib import Path

class TableLinesAndIconsRemover:

    def __init__(self, image, image_path, icon_dict_1, icon_dict_2):
        self.image = image
        self.image_path = image_path
        self.color_limit_1 = icon_dict_1
        self.color_limit_2 = icon_dict_2
    
    def execute(self):
        self.mask_icons()
        self.store_process_image("0_icon_color_masked.jpg", self.icon_color_masked_image)
        self.grayscale_image()
        self.store_process_image("1_grayscaled.jpg", self.grey)
        self.threshold_image()
        self.store_process_image("2_thresholded.jpg", self.thresholded_image)
        self.invert_image()
        self.store_process_image("3_inverted.jpg", self.inverted_image)
        self.erode_vertical_lines()
        self.store_process_image("4_erode_vertical_lines.jpg", self.vertical_lines_eroded_image)
        self.erode_horizontal_lines()
        self.store_process_image("5_erode_horizontal_lines.jpg", self.horizontal_lines_eroded_image)
        self.erode_icons()
        self.store_process_image("6_erode_icons.jpg", self.icons_eroded_image)
        self.combine_eroded_images()
        self.store_process_image("7_combined_eroded_images.jpg", self.combined_image)
        self.dilate_combined_image_to_make_lines_thicker()
        self.store_process_image("8_dilated_combined_image.jpg", self.combined_image_dilated)
        self.subtract_combined_and_dilated_image_from_original_image()
        self.store_process_image("9_image_without_lines.jpg", self.image_without_lines)
        self.remove_noise_with_erode_and_dilate()
        self.store_process_image("10_image_without_lines_noise_removed.jpg", self.image_without_lines_noise_removed)
        return self.image_without_lines_noise_removed

    def mask_icons(self):
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
        self.icon_color_masked_image = cv2.bitwise_and(self.image, self.image, mask=mask)

    def grayscale_image(self):
        self.grey = cv2.cvtColor(self.icon_color_masked_image, cv2.COLOR_BGR2GRAY)

    # playing with the pixel threshold value enables to deal with the shades at the intersection of the book
    def threshold_image(self):
        # self.thresholded_image = cv2.threshold(self.grey, 155, 255, cv2.THRESH_BINARY)[1]
        self.thresholded_image = cv2.threshold(self.grey, 125, 255, cv2.THRESH_BINARY)[1]

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)
    
    def erode_vertical_lines(self):
        # Structural element is an horizontal line, so that erosion removes vertical lines
        # as pixels on the right or left of the vertical line are black (value of 0)
        # Size of the line was adapted to also remove the icons
        hor = np.array([[1,1,1,1,1,1]])
        self.vertical_lines_eroded_image = cv2.erode(self.inverted_image, hor, iterations=10)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, hor, iterations=10)
    
    def erode_horizontal_lines(self):
        # Structural element is a vertical line, so that erosion removes vertical lines
        # as pixels above or below of the horizontal line are black (value of 0)
        # Size of the line was adapted to also remove the icons
        ver = np.array([[1],
                [1],
                [1],
                [1],
                [1],
                [1]])
        self.horizontal_lines_eroded_image = cv2.erode(self.inverted_image, ver, iterations=10)
        self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, ver, iterations=10)
    
    def erode_icons(self):
        # Structural element is a diamond, so that erosion removes icons
        # as pixels above or below of the horizontal line are black (value of 0)
        # Size of the line was adapted to also remove the icons
        ver = np.array([[1,1,1],
                [1,1,1],        
                [1,1,1],
                ])
        self.icons_eroded_image = cv2.erode(self.inverted_image, ver, iterations=10)
        self.icons_eroded_image = cv2.dilate(self.icons_eroded_image, ver, iterations=10)

    def combine_eroded_images(self):
        self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)
        self.combined_image = cv2.add(self.combined_image, self.icons_eroded_image)
    
    def dilate_combined_image_to_make_lines_thicker(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.combined_image_dilated = cv2.dilate(self.combined_image, kernel, iterations=5)
    
    def subtract_combined_and_dilated_image_from_original_image(self):
        self.image_without_lines = cv2.subtract(self.inverted_image, self.combined_image_dilated)
    
    def remove_noise_with_erode_and_dilate(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.image_without_lines_noise_removed = cv2.erode(self.image_without_lines, kernel, iterations=1)
        self.image_without_lines_noise_removed = cv2.dilate(self.image_without_lines_noise_removed, kernel, iterations=1)
    
    def store_process_image(self, file_name, image):
        path = Path("images/debug/lines_removal/" + self.image_path.stem + "_" + file_name)
        cv2.imwrite(str(path.resolve()), image) #OpenCV cannot handle Path objects, it expects strings
