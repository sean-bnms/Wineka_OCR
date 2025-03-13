from dataclasses import dataclass, field
from enum import StrEnum, auto

# allows modules to access modules from outside the package
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

# import modules from the project
import image_processing as image_processing
from cv_operations.MorphologicalTransformer import MorphologicalTransformer, MorphologicalOperation
from cv_operations.ImagePreProcessor import ImagePreProcessor, GlobalThresholder


class BoundingBoxExtractionState(StrEnum):
    BINARY_REPRESENTATION = auto()
    TEXT_DILATION = auto()
    BLOBS = auto()
    ALL_BOUNDING_BOXES = auto()
    UNWANTED_BOUNDING_BOXES_1 = auto()
    CORRECTED_BOUNDING_BOXES = auto()
    UNWANTED_BOUNDING_BOXES_2 = auto()
    ALL_INCORRECT_BOUNDING_BOXES = auto()
    FINAL_BOUNDING_BOXES = auto()
    

type BoundingBox = tuple[int, int, int, int]

@dataclass
class TextBoundingBoxExtractor:
    '''
    Extracts all the bounding box containg the text to extract from the table, organized by columns
    - image: the mathematical representation of the binary image we want to extract the text from, with the text in white and the lines already removed
    - original_image: the mathematical representation of the image with the table extracted without processing, for debugging purposes
    '''
    image: image_processing.Image
    original_image: image_processing.Image
    _transformation_states: list[str] = field(default_factory= lambda: [state.value for state in BoundingBoxExtractionState])

    def run(self) -> tuple[dict[str,list[BoundingBox]], list[BoundingBox]]:
        # when an image loads with imread, it loads it with 3 channels even when black and white pixels only
        # this avoids bugs during contour detection by converting to 2 channels
        image_preprocessor = ImagePreProcessor(image=self.image, thresholder=GlobalThresholder())
        self.binary_image = self.convert_to_binary_representation(image_preprocessor=image_preprocessor)
        # dilates text to be able to get the text blobs
        self.text_dilated_image = self.create_blobs(image=self.binary_image)

        # detects the blobs contours
        contours = image_processing.get_contours(image=self.text_dilated_image, collectHierarchy=True, useApproximation=True)
        # draws blobs on original image for clean debugging
        self.image_with_blobs = self.original_image.copy()
        image_processing.draw_contours(image=self.image_with_blobs, contours=contours)

        # detects the bounding boxes
        all_bounding_boxes = [image_processing.get_bounding_box(contour=contour) for contour in contours]
        # draws bounding boxes on original image for clean debugging
        self.image_with_all_bounding_boxes = self.original_image.copy()
        self.visualize_bounding_boxes(image=self.image_with_all_bounding_boxes, bounding_boxes=all_bounding_boxes)

        ### removes the bounding boxes unwanted, due to imperfect lines / icons erosion
        # computes unwanted bounding boxes
        unwanted_bounding_boxes_1 = self.get_unwanted_bounding_boxes(all_bounding_boxes=all_bounding_boxes)
        self.image_with_unwanted_bounding_boxes_1 = self.original_image.copy()
        self.visualize_bounding_boxes(image=self.image_with_unwanted_bounding_boxes_1, bounding_boxes=unwanted_bounding_boxes_1)
        # computes updated bounding boxes
        self.image_with_corrected_bounding_boxes = self.original_image.copy()
        corrected_bounding_boxes = self.update_bounding_boxes(image_with_updated_boxes=self.image_with_corrected_bounding_boxes, all_boxes=all_bounding_boxes, unwanted_boxes=unwanted_bounding_boxes_1)

        ### sorts the bounding boxes by columns
        sorted_by_x_boxes = self.sort_bounding_boxes_by_x_coordinate(bounding_boxes=corrected_bounding_boxes)
        columns = self.sort_ordered_bounding_boxes_by_columns(bounding_boxes=sorted_by_x_boxes)
        # removes the bounding boxes which result from longer lines not properly eroded which formed extra columns in the table
        table_columns, unwanted_bounding_boxes_2 = self.clean_table_columns(table_columns=columns, expected_col_number=3)
        
        ### visualize all identified incorrect boxes
        # second iteration of incorrect boxes identification
        self.image_with_unwanted_bounding_boxes_2 = self.original_image.copy()
        self.visualize_bounding_boxes(image=self.image_with_unwanted_bounding_boxes_2, bounding_boxes=unwanted_bounding_boxes_2)
        # all identified incorrect boxes
        self.image_with_all_incorrect_bounding_boxes = self.original_image.copy()
        self.visualize_bounding_boxes(image=self.image_with_all_incorrect_bounding_boxes, bounding_boxes=unwanted_bounding_boxes_1 + unwanted_bounding_boxes_2)

        ### visualize all correct boxes identified
        self.image_with_final_bounding_boxes = self.original_image.copy()
        correct_bounding_boxes = self.update_bounding_boxes(image_with_updated_boxes=self.image_with_final_bounding_boxes, all_boxes=corrected_bounding_boxes, unwanted_boxes=unwanted_bounding_boxes_2)
    
        return table_columns, correct_bounding_boxes


    def convert_to_binary_representation(self, image_preprocessor:ImagePreProcessor) -> image_processing.Image:
        return image_preprocessor.apply()

    def create_blobs(self, image:image_processing.Image) -> image_processing.Image:
        # first dilation helps creating the blobs
        first_dilation_transformer = MorphologicalTransformer(
            image=image,
            operation=MorphologicalOperation.DILATION,
            kernel=image_processing.Kernel(
                shape=image_processing.KernelShape.RECTANGLE,
                dimensions=(10,2)
            ),
            nbr_iterations=5
        ) 
        first_dilation = first_dilation_transformer.apply()
        # second dilation removes the remaining gaps inside blobs, e.g. to get accents in the main block
        second_dilation_transformer = MorphologicalTransformer(
            image=first_dilation,
            operation=MorphologicalOperation.DILATION,
            kernel=image_processing.Kernel(
                shape=image_processing.KernelShape.RECTANGLE,
                dimensions=(5,5)
            ),
            nbr_iterations=2
        ) 
        return second_dilation_transformer.apply()
    
    def get_mean_box_height(self, bounding_boxes:list[BoundingBox]) -> float:
        '''
        Get the mean height of the text bounding boxes provided.
        - bounding_boxes: list of the bounding boxes, in the format (top-left-corner_x, top-left-corner_y, box_width, box_height)
        '''
        bounding_box_heights = [box[3] for box in bounding_boxes]
        return sum(bounding_box_heights) / len(bounding_box_heights)

    def get_unwanted_bounding_boxes(self, all_bounding_boxes:list[BoundingBox]) -> list[BoundingBox]:
        mean_box_height = self.get_mean_box_height(bounding_boxes=all_bounding_boxes)
        #text boxes have more or less the same size, under this threshold it is certainly a line
        return [box for box in all_bounding_boxes if box[3] < (mean_box_height / 1.5)]
    
    def get_correct_bounding_boxes(self, all_bounding_boxes:list[BoundingBox], unwanted_bounding_boxes:list[BoundingBox]) -> list[BoundingBox]:
        all_box_set = {box for box in all_bounding_boxes}
        unwanted_box_set = {box for box in unwanted_bounding_boxes}
        correct_box_set = all_box_set.symmetric_difference(unwanted_box_set)
        return list(correct_box_set)
    
    def update_bounding_boxes(self, image_with_updated_boxes:image_processing.Image, all_boxes: list[BoundingBox], unwanted_boxes:list[BoundingBox]):
        correct_bounding_boxes = self.get_correct_bounding_boxes(all_bounding_boxes=all_boxes, unwanted_bounding_boxes=unwanted_boxes)
        self.visualize_bounding_boxes(image=image_with_updated_boxes, bounding_boxes=correct_bounding_boxes)
        return correct_bounding_boxes
         
    def visualize_bounding_boxes(self,image:image_processing.Image, bounding_boxes:list[BoundingBox]) -> None:
        '''
        Draws the bounding boxes on the image.
        - image: the mathematical representantion of the image where we want to draw the boxes
        - bounding_boxes: list of the bounding boxes, in the format (top-left-corner_x, top-left-corner_y, box_width, box_height)
        '''
        for box in bounding_boxes:
            top_left_corner = (box[0], box[1])
            bottom_right_corner = (box[0] + box[2], box[1] + box[3])
            image_processing.draw_rectangle(image=image, top_left_point=top_left_corner, bottom_right_point=bottom_right_corner)
    
    def sort_bounding_boxes_by_x_coordinate(self, bounding_boxes:list[BoundingBox]) -> list[BoundingBox]:
        '''
        Sorts the bounding boxes based on their top-left corner x coordinate.
        '''
        return sorted(bounding_boxes, key=lambda x: x[0])
    
    def sort_ordered_bounding_boxes_by_columns(self, bounding_boxes:list[BoundingBox]) -> dict[str,list[BoundingBox]]:
        '''
        Sorts the bounding boxes by columns. 
        - bounding_boxes: the list of bounding boxes, ordered by x coordinates
        '''
        # initialization
        columns = {}
        k = 1
        columns[f"{k}"] = [bounding_boxes[0]]
        mid_interval = columns[f"{k}"][0][0]
        column_x_values = [mid_interval]
        for i in range(1,len(bounding_boxes)):
            x = bounding_boxes[i][0]
            # we compute the range of x values where the top left corner of the 
            # bounding box should lie, by accounting for perspective distortion
            column_start_interval = range(mid_interval - 30, mid_interval + 30)
            # as the boxes are ordered by top-left corner x coordinates, if the x value
            # doesn't belong to the interval anymore, we reached a new column of the table
            if x in column_start_interval:
                columns[f"{k}"].append(bounding_boxes[i])
                column_x_values.append(bounding_boxes[i][0])
                #we want to have a centered mid_interval, as we have a tendency to see x increase for a column because of the perspective distortion
                mid_interval = sum(column_x_values) // len(column_x_values) 
            else:
                k+=1
                columns[f"{k}"] = [bounding_boxes[i]]
                column_x_values = [bounding_boxes[i][0]]
                mid_interval = columns[f"{k}"][0][0]
        return columns
    
    def clean_table_columns(self, table_columns:dict[str,list[BoundingBox]], expected_col_number:int) -> tuple[dict[str,list[BoundingBox]],list[BoundingBox]]:
        '''
        Removes from the columns storing the bounding boxes of the table the ones which can be associated with
        bigger portions of table lines which were not completely eroded. It also provides the nex incorrect boxes detected.
        - table_columns: a dictionary containing as keys the number of the columns, from left to right on the image,
        and as values a list of the bounding boxes which were found in the column
        - expected_col_number: number of columns expected for the table
        '''
        incorrect_boxes = []
        columns = table_columns.copy()
        columns_length = {k : len(columns[k]) for k in columns.keys()}
        if len(columns_length) > expected_col_number: 
            while len(columns_length) != expected_col_number:
                # removes the column which has the less boundary boxes
                min_key = min(columns_length, key = columns_length.get)
                incorrect_boxes += columns[min_key]
                del columns_length[min_key]
                del columns[min_key]
        return columns, incorrect_boxes
    
    
    ### TRANSFORMATION STATES HANDLING

    def get_transformation_states_mapping(self):
        return {
            BoundingBoxExtractionState.BINARY_REPRESENTATION: self.binary_image,
            BoundingBoxExtractionState.TEXT_DILATION: self.text_dilated_image,
            BoundingBoxExtractionState.BLOBS: self.image_with_blobs,
            BoundingBoxExtractionState.ALL_BOUNDING_BOXES: self.image_with_all_bounding_boxes,
            BoundingBoxExtractionState.UNWANTED_BOUNDING_BOXES_1: self.image_with_unwanted_bounding_boxes_1,
            BoundingBoxExtractionState.CORRECTED_BOUNDING_BOXES: self.image_with_corrected_bounding_boxes,
            BoundingBoxExtractionState.UNWANTED_BOUNDING_BOXES_2: self.image_with_unwanted_bounding_boxes_2,
            BoundingBoxExtractionState.ALL_INCORRECT_BOUNDING_BOXES: self.image_with_all_incorrect_bounding_boxes,
            BoundingBoxExtractionState.FINAL_BOUNDING_BOXES: self.image_with_final_bounding_boxes
        }
    
    def get_transformation_states(self):
        return self._transformation_states


def main():
    img_handler = image_processing.ImageHandler(image_path="images/debug/IMG_0148_without_lines.jpg")
    image = img_handler.load_image()

    original_img_handler = image_processing.ImageHandler(image_path="images/debug/IMG_0148_extracted_table.jpg")
    original_image = original_img_handler.load_image()

    bounding_box_extractor = TextBoundingBoxExtractor(
        image=image,
        original_image=original_image
    )
    table_columns, bounding_boxes = bounding_box_extractor.run()

    folder_path = "images/debug/"
    img_path_1 = img_handler.store_image(file_name="box_extraction.jpg", folder_path=folder_path, image=bounding_box_extractor.image_with_final_bounding_boxes)
    img_path_2 = img_handler.store_debug_image(
        folder_path=folder_path,
        state_mapping=bounding_box_extractor.get_transformation_states_mapping(),
        states=bounding_box_extractor.get_transformation_states(),
        state=BoundingBoxExtractionState.ALL_INCORRECT_BOUNDING_BOXES
        )
    print(img_path_1)
    print(img_path_2)
    print(bounding_boxes)
    print(table_columns)
  



if __name__ == "__main__":
    main()