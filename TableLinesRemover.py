from dataclasses import dataclass, field
from enum import StrEnum, auto

import image_processing
from MorphologicalTransformer import MorphologicalTransformer, MorphologicalOperation
from ImagePreProcessor import ImagePreProcessor, GlobalThresholder

class LinesRemovingState(StrEnum):
    BINARY_REPRESENTATION = auto()
    HORIZONTAL_LINES_EROSION = auto()
    HORIZONTAL_LINES_DILATION = auto()
    VERTICAL_LINES_EROSION = auto()
    VERTICAL_LINES_DILATION = auto()
    ALL_LINES_DILATION = auto()


@dataclass
class TableLinesRemover:
    '''
    Removes lines from the black and white image (lines are represented in white).
    - image: the mathematical representation of the source image
    - vertical_lines_kernel: contains the kernel used to remove vertical lines from the table
    - horizontal_lines_kernel: contains the kernel used to remove horizontal lines from the table
    '''
    image: image_processing.Image
    vertical_lines_kernel: image_processing.Kernel
    horizontal_lines_kernel: image_processing.Kernel
    _transformation_states: list[str] = field(default_factory= lambda: [state.value for state in LinesRemovingState])

    def run(self) -> image_processing.Image: 
        '''
        Removes both vertical and horizontal lines from the image.
        '''
        # when an image loads with imread, it loads it with 3 channels even when black and white pixels only
        # this avoids bugs later by converting to 2 channels
        image_preprocessor = ImagePreProcessor(image=self.image, thresholder=GlobalThresholder())
        self.binary_image = self.convert_to_binary_representation(image_preprocessor=image_preprocessor)
        ### horizontal lines
        # erosion
        h_erosion_transformer = MorphologicalTransformer(
            image=self.binary_image, 
            operation=MorphologicalOperation.EROSION, 
            kernel=self.horizontal_lines_kernel,
            nbr_iterations=10) # attribute value found by experimenting with the images
        self.horizontally_eroded_image = self.erode_lines(erosion_transformer=h_erosion_transformer)
        # following a dilation to ensure all pixels are considered
        h_dilation_transformer = MorphologicalTransformer(
            image=self.horizontally_eroded_image, 
            operation=MorphologicalOperation.DILATION, 
            kernel=self.horizontal_lines_kernel,
            nbr_iterations=10)
        self.horizontally_dilated_image = self.dilate_lines(dilation_transformer=h_dilation_transformer)
        
        ### vertical lines
        # erosion
        v_erosion_transformer = MorphologicalTransformer(
            image=self.binary_image, 
            operation=MorphologicalOperation.EROSION, 
            kernel=self.vertical_lines_kernel,
            nbr_iterations=10)
        self.vertically_eroded_image = self.erode_lines(erosion_transformer=v_erosion_transformer)
        # dilation
        v_dilation_transformer = MorphologicalTransformer(
            image=self.vertically_eroded_image, 
            operation=MorphologicalOperation.DILATION, 
            kernel=self.vertical_lines_kernel,
            nbr_iterations=10)
        self.vertically_dilated_image = self.dilate_lines(dilation_transformer=v_dilation_transformer)
        
        ### remove lines from image
        combined_extracted_lines_image = self.combine_lines_dilations()
        # We apply a final dilation to thicken the lines
        dilation_kernel = image_processing.Kernel(
            shape=image_processing.KernelShape.RECTANGLE,
            dimensions=(5,5)
        )
        dilation_transformer = MorphologicalTransformer(
            image=combined_extracted_lines_image, 
            operation=MorphologicalOperation.DILATION, 
            kernel=dilation_kernel)
        self.all_dilated_lines_image = self.dilate_lines(dilation_transformer=dilation_transformer)
        return self.subtract_lines_from_original_image(image=self.all_dilated_lines_image)
    
    def convert_to_binary_representation(self, image_preprocessor:ImagePreProcessor) -> image_processing.Image:
        return image_preprocessor.apply()
    
    def erode_lines(self, erosion_transformer:MorphologicalTransformer) -> image_processing.Image:
        return erosion_transformer.apply()
    
    def dilate_lines(self, dilation_transformer:MorphologicalTransformer) -> image_processing.Image:
        return dilation_transformer.apply()
    
    def combine_lines_dilations(self) -> image_processing.Image:
        return image_processing.add_images(image1=self.horizontally_dilated_image, image2=self.vertically_dilated_image)
    
    def subtract_lines_from_original_image(self, image:image_processing.Image) -> image_processing.Image:
        '''
        Removes the identified lines from the intial image.
        - image: the image containing a black background and all the dilated lines from the original image
        '''
        # substraction is applied on the inverted image, as morphology operations were applied from it
        return image_processing.substract_images(image1=self.binary_image, image2=image)
    
    ### TRANSFORMATION STATES HANDLING

    def get_transformation_states_mapping(self):
        return {
            LinesRemovingState.BINARY_REPRESENTATION: self.binary_image,
            LinesRemovingState.HORIZONTAL_LINES_EROSION: self.horizontally_eroded_image,
            LinesRemovingState.HORIZONTAL_LINES_DILATION: self.horizontally_dilated_image,
            LinesRemovingState.VERTICAL_LINES_EROSION: self.vertically_eroded_image,
            LinesRemovingState.VERTICAL_LINES_DILATION: self.vertically_dilated_image,
            LinesRemovingState.ALL_LINES_DILATION: self.all_dilated_lines_image
        }
    
    def get_transformation_states(self):
        return self._transformation_states

def main():
    img_handler = image_processing.ImageHandler(image_path="images/debug/IMG_0148_image_without_icons.jpg")
    image = img_handler.load_image()
    
    table_lines_remover = TableLinesRemover(
        image=image,
        vertical_lines_kernel= image_processing.Kernel(
            shape=image_processing.KernelShape.RECTANGLE,
            dimensions=(1,6)
        ),
        horizontal_lines_kernel= image_processing.Kernel(
            shape=image_processing.KernelShape.RECTANGLE,
            dimensions=(6,1)
        )
    )
    image_without_lines = table_lines_remover.run()

    folder_path = "images/debug/"
    img_path_1 = img_handler.store_image(file_name="and_without_lines.jpg", folder_path=folder_path, image=image_without_lines)
    img_path_2 = img_handler.store_debug_image(
        folder_path=folder_path,
        state_mapping=table_lines_remover.get_transformation_states_mapping(),
        states=table_lines_remover.get_transformation_states(),
        state=LinesRemovingState.ALL_LINES_DILATION
        )
    print(img_path_1)
    print(img_path_2)


if __name__ == "__main__":
    main()