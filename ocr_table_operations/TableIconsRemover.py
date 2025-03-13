from dataclasses import dataclass, field
from enum import StrEnum, auto

# allows modules to access modules from outside the package
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

# import modules from the project
import image_processing as image_processing
from cv_operations.ColorFilter import ColorFilter, Color
from cv_operations.ImagePreProcessor import ImagePreProcessor, Thresholder, GlobalThresholder
from cv_operations.MorphologicalTransformer import MorphologicalTransformer, MorphologicalOperation



class IconRemovingState(StrEnum):
    ICONS_FILTERING = auto()
    BINARY_REPRESENTATION = auto()
    ICONS_EROSION = auto()
    ICONS_DILATION = auto()

@dataclass
class TableIconsRemover:
    '''
    Removes icons with a specific color from an image.
    - image: the mathematical representation of the image of the (extracted) table
    - threshold: the thresholding method used to obtain a binary colored image
    - icon_colors: the list of RGB color of the icons we want to remove from the image
    '''
    image: image_processing.Image
    thresholder: Thresholder
    icon_colors: list[tuple[int, int, int]]
    _transformation_states: list[str] = field(default_factory= lambda: [state.value for state in IconRemovingState])

    def run(self) -> image_processing.Image: 
        '''
        Removes the icons from the image
        '''
        hsv = image_processing.convert_image_to_hsv(image=self.image)
        icons_kernel = image_processing.Kernel(
            shape=image_processing.KernelShape.ELLIPSE, 
            dimensions=(20,20))
        
        # remove colors from icons and turn them to black pixels
        color_masks = [ColorFilter(color=Color(rgb_color=color), image=self.image).create_color_mask(hsv=hsv)  for color in self.icon_colors]   
        self.filtered_image = self.filter_icons(color_masks=color_masks, kernel=icons_kernel)
        image_preprocessor = ImagePreProcessor(image=self.filtered_image, thresholder=self.thresholder)
        binary_image = self.convert_to_binary_representation(image_preprocessor=image_preprocessor)
        # inversion is needed to perform dilation: as kernel shapes are created with white pixels,
        # we need the contours of the table which we want to dilate to be represented by white pixels
        self.inverted_binary_image = image_processing.invert_image(image=binary_image)
        
        # erosion of the icons
        erosion_transformer = MorphologicalTransformer(
            image=self.inverted_binary_image, 
            operation=MorphologicalOperation.EROSION, 
            kernel=icons_kernel)
        self.eroded_icons_image = self.erode_icons(erosion_transformer=erosion_transformer)
        
        # following a dilation to ensure all icons pixels are considered
        dilation_transformer = MorphologicalTransformer(
            image=self.eroded_icons_image, 
            operation=MorphologicalOperation.DILATION, 
            kernel=icons_kernel, 
            nbr_iterations=5)
        self.dilated_icons_image = self.dilate_icons(dilation_transformer=dilation_transformer)
        # remove icons pixels from image
        return self.subtract_icons_from_original_image()

    def filter_icons(self, color_masks:list[ColorFilter], kernel:image_processing.Kernel) -> image_processing.Image:
        '''
        Masks the icons from the image: icons will appear with black pixels on the image.
        - color_masks: the masks applied to perform color filtering
        - kernel: the kernel used to apply morphological transformations on the icons
        '''
        mask = sum(color_masks)
        # apply a morphology operation, closing: it results in a Dilation followed by Erosion
        # helps filling small pixel gaps in an area (e.g. holes in a shape because of different lighting conditions)
        closing_transformer = MorphologicalTransformer(
            image=mask, 
            operation=MorphologicalOperation.CLOSING,
            kernel=kernel)
        morph = closing_transformer.apply()
        # filters
        mask = 255 - morph
        return image_processing.apply_bitwise_and(image1=self.image, image2=self.image, mask=mask)
    
    def convert_to_binary_representation(self, image_preprocessor:ImagePreProcessor) -> image_processing.Image:
        return image_preprocessor.apply()
    
    def erode_icons(self, erosion_transformer:MorphologicalTransformer) -> image_processing.Image:
        return erosion_transformer.apply()
    
    def dilate_icons(self, dilation_transformer:MorphologicalTransformer) -> image_processing.Image:
        return dilation_transformer.apply()
    
    def subtract_icons_from_original_image(self) -> image_processing.Image:
        # substraction is applied on the inverted image, as morphology operations were applied from it
        return image_processing.substract_images(image1=self.inverted_binary_image, image2=self.dilated_icons_image)
    
    ### TRANSFORMATION STATES HANDLING

    def get_transformation_states_mapping(self):
        return {
            IconRemovingState.ICONS_FILTERING: self.filtered_image,
            IconRemovingState.BINARY_REPRESENTATION: self.inverted_binary_image,
            IconRemovingState.ICONS_EROSION: self.eroded_icons_image,
            IconRemovingState.ICONS_DILATION: self.dilated_icons_image,
        }
    
    def get_transformation_states(self):
        return self._transformation_states

def main():
    img_handler = image_processing.ImageHandler(image_path="images/debug/IMG_0148_otsu.jpg")
    image = img_handler.load_image()

    gold_icons_color = (158, 130, 90)
    red_icons_color = (163, 151, 152)
    # extraction with a simple thresholder
    table_icon_remover = TableIconsRemover(
        image=image,
        icon_colors=[gold_icons_color, red_icons_color],
        thresholder=GlobalThresholder()
    )
    image_without_icons = table_icon_remover.run()

    folder_path = "images/debug/"
    img_path_1 = img_handler.store_image(file_name="image_without_icons.jpg", folder_path=folder_path, image=image_without_icons)
    img_path_2 = img_handler.store_debug_image(
        folder_path=folder_path,
        state_mapping=table_icon_remover.get_transformation_states_mapping(),
        states=table_icon_remover.get_transformation_states(),
        state=IconRemovingState.ICONS_FILTERING
        )
    print(img_path_1)
    print(img_path_2)


if __name__ == "__main__":
    main()