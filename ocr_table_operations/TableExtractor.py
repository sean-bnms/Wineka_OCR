from dataclasses import dataclass, field
from enum import StrEnum, auto

# allows modules to access modules from outside the package
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

# import modules from the project
import image_processing as image_processing
from cv_operations.ColorFilter import ColorFilter, Color
from cv_operations.ImagePreProcessor import ImagePreProcessor, Thresholder, GlobalThresholder, GlobalOptimizedThresholder
from cv_operations.MorphologicalTransformer import MorphologicalTransformer, MorphologicalOperation


class TableExtractionState(StrEnum):
    PREPROCESSING = auto()
    DILATION = auto()
    ALL_CONTOURS = auto()
    TABLE_EDGES = auto()
    TABLE_EXTRACTION = auto()


@dataclass
class RectangleEdges:
    top_left: tuple[int, int]
    top_right: tuple[int, int]
    bottom_right: tuple[int, int]
    bottom_left: tuple[int, int]


@dataclass
class TableExtractor:
    '''
    Extract the table within an image.
    - image: the mathematical representation of the source image, \n
    - threshold: the thresholding method used to obtain a binary colored image, \n
    - background_color: (Optional) The RGB color of the background of the image, if its is different from the table internal color
    '''
    image: image_processing.Image
    thresholder: Thresholder
    background_color: tuple[int, int, int] | None = None
    _transformation_states: list[str] = field(default_factory= lambda: [state.value for state in TableExtractionState])

    def run(self) -> image_processing.Image:
        '''
        Extracts the table from the image provided and returns it with a white padding to ease future morphological transformations
        '''
        # preprocess image to remove color dependency
        # obtain an image with black background and white lines and characters
        self.preprocessed_image = self.preprocess_image()
        
        # apply dilation to make contours more recognizable
        dilation_transformer = MorphologicalTransformer(image=self.preprocessed_image, operation=MorphologicalOperation.DILATION)
        self.dilated_image = dilation_transformer.apply()
        
        # start contour recognition
        contours = image_processing.get_contours(image=self.dilated_image, collectHierarchy=False, useApproximation=True)
        self.image_with_all_contours = self.image.copy()
        self.visualize_contours(image=self.image_with_all_contours, contours=contours)
        
        # identify the table edges
        table_edges = self.get_table_edges(contours=contours)
        self.image_with_table_edges = self.image.copy()
        self.visualize_table_edges(image=self.image_with_table_edges, table_edges=table_edges)
        
        # extracts the table
        self.extracted_table_image = self.resize_image(table_edges=table_edges)

        # stores the transformations for debugging purposes
        self.transformation_states_mapping = self.get_transformation_states_mapping()

        # adds padding to ease line detection for future morphological operations
        return image_processing.add_padding(image=self.extracted_table_image, percentage=5)


    def preprocess_image(self) -> image_processing.Image: 
        '''
        Applies background color filtering to the image and converts it to a binary image, with black background and white lines / characters.
        '''
        if self.background_color != None:
            color_filter = ColorFilter(color=Color(rgb_color=self.background_color), image=self.image)
            self.filtered_image = self.filter_background_color(color_filter=color_filter)
            image_preprocessor = ImagePreProcessor(image=self.filtered_image, thresholder=self.thresholder)
        # no need for background color filtering
        else:
            self.filtered_image = None
            image_preprocessor = ImagePreProcessor(image=self.image, thresholder=self.thresholder)

        self.binary_image = self.convert_to_binary_representation(image_preprocessor=image_preprocessor)
        # inversion is needed to perform dilation: as kernel shapes are created with white pixels,
        # we need the contours of the table which we want to dilate to be represented by white pixels
        return image_processing.invert_image(image=self.binary_image)

    def filter_background_color(self, color_filter:ColorFilter) -> image_processing.Image:
        return color_filter.filter()
    
    def convert_to_binary_representation(self, image_preprocessor:ImagePreProcessor) -> image_processing.Image:
        return image_preprocessor.apply()
    
    def visualize_contours(self, image:image_processing.Image, contours: list) -> None:
        image_processing.draw_contours(image=image, contours=contours)
    
    # The approach to compute the contours with a rectangular shape to keep table and then pick the ones with the biggest area
    # was abandoned because it wouldn't always categorize the table contour as rectangular
    # Instead, we look for optimal coordinates of the edges of the table, then compute the real edges based on distances between point
    
    def get_contour_extremums(self, contour:list) -> tuple[int,int,int,int]:
        '''
        Computes the extremum values for a contour coordinates, along both x and y axis 
        - contour: a contour computed from the binary image analyzed
        '''
        x_values = [point[0][0] for point in contour]
        y_values = [point[0][1] for point in contour]
        x_max = max(x_values)
        y_max = max(y_values)
        x_min = min(x_values)
        y_min = min(y_values)
        return x_max, x_min, y_max, y_min
    
    def get_optimal_table_edges(self, contours:list) -> RectangleEdges:
        '''
        Computes the optimal edges of the table, if the image had no deformation.
        - contours: contours from the image, here their approximation is expected to avoid noise
        '''
        # intialize table contours
        height, width = self.image.shape[0], self.image.shape[1]
        x_max_table, x_min_table, y_max_table , y_min_table = 0, width,0, height
        for contour in contours:
            extremums = self.get_contour_extremums(contour=contour)
            # adds margin to remove the picture frame coordinates which are also detected as contours
            x_max_table = extremums[0] if extremums[0] > x_max_table and extremums[0] < width - 10 else x_max_table
            x_min_table = extremums[1] if extremums[1] < x_min_table and extremums[1] > 10 else x_min_table
            y_max_table = extremums[2] if extremums[2] > y_max_table and extremums[2] < height - 10 else y_max_table
            y_min_table = extremums[3] if extremums[3] < y_min_table and extremums[3] > 10 else y_min_table
        return RectangleEdges(
            top_left=(x_min_table, y_min_table), 
            top_right=(x_max_table, y_min_table), 
            bottom_right=(x_max_table, y_max_table), 
            bottom_left=(x_min_table, y_max_table)
            )
    
    def calculate_distance(self, point1:tuple[int, int], point2:tuple[int, int]):
        return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**0.5

    def get_closest_point(self, point: tuple[int, int], contours:list) -> tuple[int, int]:
        '''
        Calculates for a given point coordinates, which point of coordinates (x,y) from the contours detected in the image is the closest.
        - point: one of the optimal edge for the table, as we want to find the real points before extracting the table
        - contours: the list of approximated contours for the binary image
        '''
        # initialization
        closest_distance = 1000
        closest_point = (0,0)
        
        # computes closest point
        for contour in contours:
            distances = [self.calculate_distance(point1=point, point2=(contour_pt[0][0], contour_pt[0][1])) for contour_pt in contour]
            closest_pt_idx = distances.index(min(distances))
            if distances[closest_pt_idx] < closest_distance:
                closest_point = contour[closest_pt_idx]
                closest_distance = distances[closest_pt_idx]

        # returns the coordinates as a tupple as it is to be used in the RectangleEdges object attributes
        return (closest_point[0][0], closest_point[0][1])
    
    def get_table_edges(self, contours:list) -> RectangleEdges:
        # approximate contours to make it more robust to image noise
        contour_approximations = [image_processing.get_contour_approximation(contour=contours[i], eps=0.02, isContourClosed=True) for i in range(len(contours))]
        optimal_edges = self.get_optimal_table_edges(contours=contour_approximations)
        # we want to find the real table points, to account for image deformations
        return RectangleEdges(
            top_left=self.get_closest_point(point=optimal_edges.top_left, contours=contour_approximations),
            top_right=self.get_closest_point(point=optimal_edges.top_right, contours=contour_approximations),
            bottom_right=self.get_closest_point(point=optimal_edges.bottom_right, contours=contour_approximations),
            bottom_left=self.get_closest_point(point=optimal_edges.bottom_left, contours=contour_approximations)
        )
    
    def visualize_table_edges(self, image:image_processing.Image, table_edges:RectangleEdges) -> None:
        '''
        Draws on the image cercles to represent the 4 edges detected for the table with their coordinates.
        - image: the image on which we want to draw
        - edges: the coordinates calculated for the table
        '''
        def visualize(image:image_processing.Image, point:tuple[int,int]) -> None:
            text_annotation = str(point[0]) + ", " + str(point[1])
            image_processing.annotate_point(image=image, point=point, text=text_annotation)
            image_processing.draw_circle(image=image, point=point, radius=10)
        
        edges = [table_edges.top_left, table_edges.top_right, table_edges.bottom_right, table_edges.bottom_left]
        for edge in edges:
            visualize(image=image, point=edge)

    def get_resized_image_dimensions(self, table_edges:RectangleEdges) -> tuple[int,int]:
        image_width = self.image.shape[1]
        image_width_reduced_by_10_percent = int(image_width * 0.9)
        table_width = self.calculate_distance(point1=table_edges.top_left, point2=table_edges.top_right)
        table_height = self.calculate_distance(point1=table_edges.top_left, point2=table_edges.bottom_left)
        aspect_ratio = table_height / table_width
        new_image_width = image_width_reduced_by_10_percent
        new_image_height = int(new_image_width * aspect_ratio)
        return new_image_width, new_image_height

    def resize_image(self, table_edges:RectangleEdges) -> image_processing.Image:
        '''
        Extracts the table from the original image by applying a perspective transformation on it.
        - table_edges: the coordinates of the table corner edges calculated via contouring
        '''
        final_image_dimensions = self.get_resized_image_dimensions(table_edges=table_edges)
        table_corner_edges = [table_edges.top_left, table_edges.top_right, table_edges.bottom_right, table_edges.bottom_left]
        return image_processing.apply_perspective_transformation(image=self.image, table_corner_edges=table_corner_edges, final_image_dimensions=final_image_dimensions)
    
    ### TRANSFORMATION STATES HANDLING

    def get_transformation_states_mapping(self):
        return {
            TableExtractionState.PREPROCESSING: self.preprocessed_image,
            TableExtractionState.DILATION: self.dilated_image,
            TableExtractionState.ALL_CONTOURS: self.image_with_all_contours,
            TableExtractionState.TABLE_EDGES: self.image_with_table_edges,
            TableExtractionState.TABLE_EXTRACTION: self.extracted_table_image
        }
    
    def get_transformation_states(self):
        return self._transformation_states


def main():
    img_handler = image_processing.ImageHandler(image_path="images/IMG_0148.jpg")
    image = img_handler.load_image()

    # extraction with a simple thresholder
    table_extractor_1 = TableExtractor(
        image=image,
        background_color=(163, 151, 152),
        thresholder=GlobalThresholder()
    )
    simple_threshold_extraction = table_extractor_1.run()
    # extraction with a global thresholder optimized with Otsu method
    table_extractor_2 = TableExtractor(
        image=image,
        background_color=(163, 151, 152),
        thresholder=GlobalOptimizedThresholder()
    )
    otsu_extraction = table_extractor_2.run()

    # store the results for comparison of thresholding methods
    folder_path = "images/debug/"
    img_path_1 = img_handler.store_image(file_name="simple_threshold.jpg", folder_path=folder_path, image=simple_threshold_extraction)
    img_path_2 = img_handler.store_image(file_name="otsu.jpg", folder_path=folder_path, image=otsu_extraction)
    img_path_3 = img_handler.store_debug_image(
        folder_path=folder_path,
        state_mapping=table_extractor_2.get_transformation_states_mapping(),
        states=table_extractor_2.get_transformation_states(),
        state=TableExtractionState.TABLE_EDGES
        )
    print(img_path_1)
    print(img_path_2)
    print(img_path_3)


if __name__ == "__main__":
    main()