from dataclasses import dataclass
from enum import StrEnum, auto, Enum
from pathlib import Path

import cv2
import numpy as np

type Image = np.ndarray

class KernelShape(StrEnum):
    RECTANGLE = auto()
    ELLIPSE = auto()
    CROSS = auto()

@dataclass
class Kernel:
    '''
    Generates the kernel to be used for morphological operations.

    shape: the shape of the kernel we want to apply, accepted values are rectangle, cross or ellipse,\n
    dimensions: the dimensions of the kernel
    '''
    shape: KernelShape
    dimensions: tuple[int, int]

    # more info on how the kernel get calculated can be found here: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    def generate(self):
        match self.shape:
            case KernelShape.RECTANGLE:
                shape_opencv = cv2.MORPH_RECT
            case KernelShape.CROSS:
                shape_opencv = cv2.MORPH_CROSS
            case KernelShape.ELLIPSE:
                shape_opencv = cv2.MORPH_ELLIPSE
        return cv2.getStructuringElement(shape=shape_opencv, ksize=self.dimensions)


# .inRange associates any pixel with values lying in the range [lower_b, upper_b] to 255 (white), and the others 0
def create_mask(image:Image, boundaries: tuple[list[int], list[int]]) -> np.ndarray:
    return cv2.inRange(src=image, lowerb=np.array(boundaries[0]), upperb=np.array(boundaries[1]))


### IMAGES CONVERSION TO OTHER COLOR MODES

def convert_image_to_hsv(image:Image) -> Image:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def convert_image_from_gray_to_color(image:Image) -> Image:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def convert_image_to_grayscale(image: Image) -> Image:
    '''
    Changes the pixels representation for a given image from 3 dimensions (e.g. RGB) to 1 dimension (shades of grey, 255 being white and 0 black)
    - image: the mathematical representation of a colored image
    '''
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

### IMAGE THRESHOLDING

# Global thresholding methods

def apply_simple_binary_threshold(image: Image, threshold_value: float = 127, new_pixel_value: float = 255, inversion: bool = False) -> Image:
    '''
    Apply a simple binary threshold to the image pixel values. By default, turns every pixel above the mid range grey white.

    - image: the mathematical representation of the grayscaled version of the image we want to apply the threshold on, \n
    - threshold_value: value used for the pixel_value > threshold_value comparison, \n
    - new_pixel_value: the non null pixel value which pixels will take depending on the threshold comparisons results, \n
    - inversion: if False, new_pixel_value is assigned to pixels whose value is above the threshold, if True pixels above the threshold take the value 0 (black pixel)
    '''
    threshold_type = cv2.THRESH_BINARY_INV if inversion else cv2.THRESH_BINARY
    return cv2.threshold(src=image, thresh=threshold_value, maxval=new_pixel_value, type=threshold_type)[1]

def apply_simple_truncation_threshold(image: Image, threshold_value: float = 127) -> Image:
    '''
    Apply a simple truncation threshold to the image pixel values, meaning all pixels with values above the threshold will be assigned the threshold value.
    Pixel values under the threshold remains the same.

    - image: the mathematical representation of the grayscaled version of the image we want to apply the threshold on, \n
    - threshold_value: value used for the pixel_value > threshold_value comparison
    '''
    return cv2.threshold(src=image, thresh=threshold_value, maxval=255, type=cv2.THRESH_TRUNC)[1]

def apply_complex_binary_threshold(image: Image, new_pixel_value: int = 255, inversion: bool = False) -> Image:
    '''
    Changes the pixels representation of a grayscaled image to a binary representation (either black with 0 or white with 255) 
    based on an optimal threshold calculated via Otsu optimization method

    - image: the mathematical representation of the grayscaled version of an image, \n
    - new_pixel_value: the non null pixel value which pixels will take depending on the threshold comparisons results \n
    - inversion: if False, new_pixel_value is assigned to pixels whose value is above the threshold, if True pixels above the threshold take the value 0 (black pixel)
    '''
    threshold_type = cv2.THRESH_BINARY_INV if inversion else cv2.THRESH_BINARY
    # threshold will be computed via Otsu method, so the value provided is arbitrary
    return cv2.threshold(src=image, thresh=0, maxval=new_pixel_value, type=threshold_type + cv2.THRESH_OTSU)[1]

# Adaptive thresholding methods

class MeanMethod(Enum):
    '''
    Contains the accepted methods for Adaptive Thresholding. In OpenCV, these methods are assigned to an integer.
    '''
    GAUSSIAN = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    MEAN = cv2.ADAPTIVE_THRESH_MEAN_C

@dataclass
class AdaptiveThresholdMethod:
    '''
    Stores the parameters used for the adaptive threshold method used

    - method: the type of method used: in the MEAN value, all pixels in the surrounding neighborhood accounts for the same weigh in the comuputation of the threshold, while in GAUSSIAN they are weighted, \n
    - neighboor_matrix_size: the size of the square matrix defining the neighborhood of the pixel, an uneven value is expected as the pixel lies in the center of the matrix
    - constant: constant substracted from the mean computed with the adaptive_method
    '''
    method: MeanMethod
    neighboor_matrix_size: int
    constant: int

def apply_adaptive_threshold(image: Image, method: AdaptiveThresholdMethod, new_pixel_value: float = 255, inversion: bool = False) -> Image:
    '''
    Apply an adaptive method to threshold the image pixel values, where thresholds are calculated for each pixel based on its neighbors pixel values: 
    it is useful when the lighting conditions / shadowing vary in the picture.

    - image: the mathematical representation of the grayscaled version of the image we want to apply the threshold on, \n
    - method: the adaptive method used to compute the threshold, \n
    - new_pixel_value: the non null pixel value which pixels will take depending on the threshold comparisons results \n
    - inversion: if False, new_pixel_value is assigned to pixels whose value is above the threshold, if True pixels above the threshold take the value 0 (black pixel)
    '''
    threshold_type = cv2.THRESH_BINARY_INV if inversion else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(src=image, maxValue=new_pixel_value, thresholdType=threshold_type, adaptiveMethod=method.method.value, blockSize=method.neighboor_matrix_size, C=method.constant)


### IMAGE OPERATIONS 

def add_images(image1:Image, image2:Image) -> Image:
    return cv2.add(src1=image1, src2=image2)

def substract_images(image1:Image, image2:Image) -> Image:
    return cv2.subtract(src1=image1, src2=image2)

def invert_image(image: Image):
    '''
    Inverts the color distribution of black and white images (black pixels become white and respectively)

    - image: thresholded version of the image we analyze
    '''
    return cv2.bitwise_not(image)

def apply_bitwise_and(image1:Image, image2:Image, mask:np.ndarray):
    return cv2.bitwise_and(src1=image1, src2=image2, mask=mask)

def add_padding(image:Image, percentage:int, color: list[int,int,int] = [255, 255, 255]) -> Image:
    '''
    Adds a regular padding to the image, by default the padding color is white
    - image: the mathematical representation of the timage we want to add padding to
    - percentage: the percentage of the height we want to have for the padding size
    - color: the BGR color of the pixels forming the padding added, as a list
    '''
    image_height = image.shape[0]
    image_width = image.shape[1]
    vertical_padding = int(image_height * (percentage/100))
    horizontal_padding = int(image_width * (percentage/100))
    return cv2.copyMakeBorder(src=image, top=vertical_padding, bottom=vertical_padding, left=horizontal_padding, right=horizontal_padding, borderType=cv2.BORDER_CONSTANT, value=color)

def crop_image(image:Image, height_boundaries:tuple[int,int], width_boundaries:tuple[int,int]) -> Image:
    '''
    Returns the part of the image contained in the square formed by the x and y boundaries
    - image: the mathematical representation of the image we want to crop
    - height_boundaries: the range we want to crop vertically
    - width_boundaries: the range we want to crop horizontally
    '''
    return image[height_boundaries[0]:height_boundaries[1], width_boundaries[0]:width_boundaries[1]]

### PERSPECTIVE TRANSFORMATION
def apply_perspective_transformation(image:Image, table_corner_edges:list[tuple[int,int]], final_image_dimensions:tuple[int,int]) -> Image:
    '''
    Applies perspective transformation to an image to straigthen the lines on it.
    - image: the mathematical representation of the image we want to apply perspective transformation
    - table_corner_edges: a list containing the coordinates (x,y) of each corner points of the main table in the image, in a clockwise order starting from top left
    - final_image_dimensions: the (width, height) of the new image computed
    '''
    final_width = final_image_dimensions[0]
    final_height = final_image_dimensions[1]
    pts1 = np.float32([[table_corner_edges[i][0], table_corner_edges[i][1]] for i in range(len(table_corner_edges))])
    pts2 = np.float32([[0, 0], [final_width, 0], [final_width, final_height], [0, final_height]])
    transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(src=image, M=transformation_matrix, dsize=(final_width, final_height))


### MORPHOLOGICAL OPERATIONS

def close(image:Image, kernel:Kernel, nbr_iterations:int) -> Image:
    return cv2.morphologyEx(src=image, op=cv2.MORPH_CLOSE, kernel=kernel.generate(), iterations=nbr_iterations)

def open(image:Image, kernel:Kernel, nbr_iterations:int) -> Image:
    return cv2.morphologyEx(src=image, op=cv2.MORPH_OPEN, kernel=kernel.generate(), iterations=nbr_iterations)

def dilate(image: Image, kernel:Kernel, nbr_iterations:int) -> Image:
    return cv2.dilate(src=image, kernel=kernel.generate(), iterations=nbr_iterations)

def erode(image: Image, kernel:Kernel, nbr_iterations:int) -> Image:
    return cv2.erode(src=image, kernel=kernel.generate(), iterations=nbr_iterations)


### CONTOUR OPERATIONS

def get_contours(image:Image, collectHierarchy: bool = False ,useApproximation: bool = True) -> list[np.ndarray]:
    '''
    Get list of all contours detected in the image.

    - image: the mathematical representation of the binary image which contains the contours to detect. 
    Contours to detect need to be represented by white pixels, the image background by back pixels.
    - useApproximation: defines whether we want to approximate contours to its edges or not. If what is to be detected is composed of 
    lines, useApproximation should use the defaulted value for memory optimization.
    '''
    computation_method = cv2.CHAIN_APPROX_SIMPLE if useApproximation else cv2.CHAIN_APPROX_NONE
    hierarchy_mode = cv2.RETR_TREE if collectHierarchy else cv2.RETR_LIST
    contours, hierarchy = cv2.findContours(image=image, mode=hierarchy_mode, method=computation_method)
    return contours
        
def draw_contours(image:Image, contours: list[np.ndarray], color: tuple[int,int,int] = (0, 255, 0), thickness:int = 3, index:int = -1) -> Image:
    '''
    Adds the contours detected on top of the given image to visualize them.
    - image: the mathematical representation of the original image from which we are trying to detect contours
    - contours: a list containing all (x,y) coordonates of the contours in the image
    - color: the BGR color used for drawing the contour, by default it is green
    - thickness: the thickness of the contour, defaulted to 3
    - index: the index of the contour we want to draw from the countours list. By default, set to -1 to draw all contours contained in the list
    '''
    cnt = contours if index == -1 else [contours[index]]
    cv2.drawContours(image=image, contours=cnt, contourIdx=index, color=color, thickness=thickness)

def get_contour_perimeter(contour:np.ndarray, isContourClosed: bool) -> float:
    return cv2.arcLength(curve=contour, closed=isContourClosed)

def get_contour_area(contour:np.ndarray) -> float:
    return cv2.contourArea(contour=contour)

def get_contour_approximation(contour:np.ndarray, eps: float, isContourClosed) -> np.ndarray:
    '''
    Gets an approximation of the shape of the contour, to account for image noise.
    - contour: the mathematical representation of a shape contour, which is an array of (x,y) coordinates
    - eps: percentage of the perimeter of the contour we use for the approximation
    '''
    epsilon = eps*cv2.arcLength(curve=contour,closed=isContourClosed)
    return cv2.approxPolyDP(curve=contour,epsilon=epsilon,closed=isContourClosed)

def get_bounding_box(contour:np.ndarray) -> tuple[int,int,int,int]:
    '''
    Returns the values needed to compute the smallest box which contains the contour. The 2 first arguments are the coordinates
    (x,y) of the top left corner of the rectangle, the two next are the width and height.
    - contour: the mathematical representation of a shape contour, which is an array of (x,y) coordinates
    '''
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h 
    

### ANNOTATION OPERATIONS
def annotate_point(image:Image, point:tuple[int,int], text:str, color:tuple[int,int,int]=(255, 0, 0)) -> None:
    '''
    Adds text next to the point coordinates.
    - image: the mathematical representation of the image we want to annotate 
    - point: the x,y coordinates of the point 
    - text: text to write
    - color: color of the text, defaulted to blue
    '''
    cv2.putText(img=image , text=text, org=point, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=color) 
    
def draw_circle(image:Image, point:tuple[int,int], radius:int ,color:tuple[int,int,int]=(255, 0, 0)) -> None: 
    '''
    Adds a circle at the point coordinates.
    - image: the mathematical representation of the image we want to annotate
    - point: the x,y coordonates of the point
    - radius: radius in px of the circle to draw 
    - color: color of the circle, defaulted to blue
    '''
    # thickness is set to -1 to get the circle filled with the color passed  
    cv2.circle(img=image , center=point, radius=radius, color=color, thickness=-1)    

def draw_rectangle(image:Image, top_left_point:tuple[int,int], bottom_right_point:tuple[int,int], color:tuple[int,int,int]=(0, 255, 0), thickness:int = 5):
    '''
    Adds a rectangle at the point coordinates.
    - image: the mathematical representation of the image we want to annotate
    - top_left_point: the (x,y) coordonates of the point corresponding to the top left corner of the rectangle
    - bottom_right_point: the (x,y) coordonates of the point corresponding to the bottom right corner of the rectangle
    - color: color of the circle, defaulted to green
    - thickness: thickness of the rectangle, defaulted to 5 px
    '''
    cv2.rectangle(img=image, pt1=top_left_point, pt2=bottom_right_point, color=color, thickness=thickness)

### READING AND STORING IMAGES

@dataclass
class ImageHandler:
    '''
    Handles the loading and the storing of images
    - image_path: the path to the stored image
    '''
    image_path: str

    def load_image(self) -> Image:
        '''
        Loads the image stored at the path given
        '''
        img_path = Path(self.image_path) 
        return self._read_image(image_path=img_path)

    def _read_image(self, image_path:Path) -> Image:
        return cv2.imread(filename=str(image_path.resolve())) #OpenCV cannot handle Path objects, it expects strings

    def store_image(self, folder_path:str, file_name:str, image:Image) -> Path:
        '''
        Stores locally the image provided and returns its path.
        - folder_path: the path to the folder where the image should be stored
        - file_name: the name of the image file, along with its extension (.jpg or .png)
        - image: the image to save
        '''
        path = Path(folder_path + self.get_image_name() + "_" + file_name)
        cv2.imwrite(filename=str(path.resolve()), img=image) #OpenCV cannot handle Path objects, it expects strings
        return str(path.resolve())

    def store_debug_image(self, folder_path:str, state_mapping: dict[str,Image], states:list[str], state:str) -> str:
        '''
        Stores locally an image corresponding to a specific state of the transformation occuring.
        - folder_path: the path to the folder where the image should be stored
        - state_mapping: maps the allowed state transformation values to the corresponding transformed image
        - states: list of the allowed state transformation values
        - state: the state we want to debug
        '''
        if state not in states:
                raise ValueError(f"The state value provided is not allowed, use one of these values instead: {", ".join(states)}")

        path = self.store_image(file_name=f"{state}.jpg", folder_path=folder_path, image=state_mapping[state])
        return path
    
    def get_image_name(self) -> str:
        return Path(self.image_path).stem


def main():
    # r_bckgd, g_bckgd, b_bckgd = 163, 151, 152
    # h_bckgd, s_bckgd, v_bckgd = rgb_to_hsv(r_bckgd,g_bckgd,b_bckgd)
    # h_bckgd_n, s_bckgd_n, v_bckgd_n = normalize_hsv_for_opencv(h=h_bckgd, s=s_bckgd, v=v_bckgd)
    # r_gold, g_gold, b_gold = 158, 130, 90
    # h_gold, s_gold, v_gold = rgb_to_hsv(r_gold,g_gold,b_gold)
    # h_gold_n, s_gold_n, v_gold_n = normalize_hsv_for_opencv(h=h_gold, s=s_gold, v=v_gold)
    # print("bckgd h", h_bckgd_n)
    # print("bckgd boundaries", compute_hsv_boundaries(h_bckgd_n))
    # print("gold h", h_gold_n)
    # print("gold boundaries", compute_hsv_boundaries(h_gold_n))

    path_image = Path("images/IMG_0148.jpg")
    image = cv2.imread(str(path_image.resolve()))
    # filtered_image = filter_color(image=image, colors=[(r_bckgd, g_bckgd, b_bckgd), (r_gold, g_gold, b_gold)])
    # store_process_image(file_name="test_filtered_colors.jpg", image=filtered_image, image_path=path_image)


if __name__ == "__main__":
    main()