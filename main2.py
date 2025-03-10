from pathlib import Path
from typing import Protocol
from enum import StrEnum

from ImagePreProcessor import GlobalOptimizedThresholder, GlobalThresholder
from TableExtractor2 import TableExtractor2, TableExtractionState
from TableIconsRemover import TableIconsRemover, IconRemovingState
from TableLinesRemover import TableLinesRemover, LinesRemovingState
from TextBoundingBoxExtractor import TextBoundingBoxExtractor, BoundingBox, BoundingBoxExtractionState
from TextBoundingSorter import TextBoundingSorter
from OcrProcessor import OrcProcessor, TesseractLanguage
from image_processing import ImageHandler, Kernel, KernelShape, Image

DEBUG_FOLDER_PATH = "images/debug/"

def extract_table(background_color: tuple[int,int,int], image: Image) -> tuple[TableExtractor2,Image]:
    # otsu optimized thresholding method showed better results
    # to perform table extraction
    table_extractor = TableExtractor2(
        image=image,
        background_color=background_color,
        thresholder=GlobalOptimizedThresholder()
    )
    image = table_extractor.run()
    return table_extractor, image 

def remove_icons(icon_colors: list[tuple[int,int,int]], image_path: str) -> tuple[TableIconsRemover,Image]:
    # we start from the stored image of the previous step to avoid channel conflicts
    # as the classes were made to work independently based on image input
    img_handler = ImageHandler(image_path=image_path)
    image = img_handler.load_image()

    table_icon_remover = TableIconsRemover(
        image=image,
        icon_colors=icon_colors,
        thresholder=GlobalThresholder()
    )
    image = table_icon_remover.run()
    return table_icon_remover, image

def remove_lines(image_path: str) -> tuple[TableLinesRemover,Image]:
    # we start from the stored image of the previous step to avoid channel conflicts
    # as the classes were made to work independently based on image input
    img_handler = ImageHandler(image_path=image_path)
    image = img_handler.load_image()

    table_lines_remover = TableLinesRemover(
        image=image,
        vertical_lines_kernel= Kernel(
            shape=KernelShape.RECTANGLE,
            dimensions=(1,6)
        ),
        horizontal_lines_kernel= Kernel(
            shape=KernelShape.RECTANGLE,
            dimensions=(6,1)
        )
    )
    image = table_lines_remover.run()
    return table_lines_remover, image

def get_bounding_boxes(image_path:str, extracted_table_path:str) -> tuple[TextBoundingBoxExtractor, dict[str,list[BoundingBox]], list[BoundingBox]]:
    # we start from the stored image of the previous step to avoid channel conflicts
    # as the classes were made to work independently based on image input
    img_handler = ImageHandler(image_path=image_path)
    image = img_handler.load_image()
    extracted_table_img_handler = ImageHandler(image_path=extracted_table_path)
    extracted_table_image = img_handler.load_image()

    bounding_box_extractor = TextBoundingBoxExtractor(
        image=image,
        original_image=extracted_table_image
    )
    table_columns, bounding_boxes = bounding_box_extractor.run()
    return bounding_box_extractor, table_columns, bounding_boxes

def perform_ocr(bounding_box_array:list, extracted_table_path:str, initial_img_path:str, slices_folder:str):
    img_handler = ImageHandler(image_path=extracted_table_path)
    image = img_handler.load_image()

    ocr_processor = OrcProcessor(
        table_bounding_box_array=bounding_box_array,
        image=image,
        original_image_path=initial_img_path,
        images_folder_path=slices_folder,
        table_column_names=["meal_name", "wine_types", "wine_appelations"],
        language=TesseractLanguage.FRENCH
    )
    return ocr_processor.run()


### Debugging

class TransformationProcessor(Protocol):
    
    def get_transformation_states_mapping(self) -> dict[StrEnum,Image]:
        ...
    
    def get_transformation_states(self) -> list[str]:
        ...


def debug_transformation_process(processor:TransformationProcessor, img_handler:ImageHandler, state:str, folder_path:str=DEBUG_FOLDER_PATH) -> str:
    return img_handler.store_debug_image(
        folder_path=folder_path,
        state_mapping=processor.get_transformation_states_mapping(),
        states=processor.get_transformation_states(),
        state=state
        )


def main():
    # path to store and load images
    
    image_path = "images/IMG_0150.jpg"
    images_folder_path = "images/ocr_slices/"

    # initialize the image handler
    img_handler = ImageHandler(image_path=image_path)
    image = img_handler.load_image()

    # some variables for the table extraction
    background_color = (163, 151, 152)
    gold_icons_color = (158, 130, 90)
    red_icons_color = (163, 151, 152)

    # table extraction
    extractor, image_with_extracted_table = extract_table(background_color=background_color, image=image)
    extracted_table_path = img_handler.store_image(folder_path=DEBUG_FOLDER_PATH, file_name="extracted_table.jpg", image=image_with_extracted_table)
    # available values are TableExtractionState enums
    extraction_step_debug_path = debug_transformation_process(
        processor=extractor, 
        img_handler=img_handler, 
        state=TableExtractionState.TABLE_EDGES)
    print("Extraction Completed: \nfinal img: " + extracted_table_path + "\ndebug img: " + extraction_step_debug_path + "\n")

    # icons removal
    icon_remover, image_without_icons = remove_icons(icon_colors=[gold_icons_color, red_icons_color], image_path=extracted_table_path)
    removed_icons_path = img_handler.store_image(folder_path=DEBUG_FOLDER_PATH, file_name="without_icons.jpg", image=image_without_icons)
    # available values are IconRemovingState enums
    ic_removal_step_debug_path = debug_transformation_process(processor=icon_remover, img_handler=img_handler, state=IconRemovingState.ICONS_FILTERING)
    print("Icons Removal Completed: \nfinal img: " + removed_icons_path + "\ndebug img: " + ic_removal_step_debug_path + "\n")
    
    # lines removal
    line_remover, image_without_lines = remove_lines(image_path=removed_icons_path)
    removed_lines_path = img_handler.store_image(folder_path=DEBUG_FOLDER_PATH, file_name="without_lines.jpg", image=image_without_lines)
    # available values are LinesRemovingState enums
    ln_removal_step_debug_path = debug_transformation_process(processor=line_remover, img_handler=img_handler, state=LinesRemovingState.ALL_LINES_DILATION)
    print("Lines Removal Completed: \nfinal img: " + removed_lines_path + "\ndebug img: " + ln_removal_step_debug_path + "\n")

    # text bounding boxes extraction
    bounding_box_extractor, table_columns, bounding_boxes = get_bounding_boxes(image_path=removed_lines_path, extracted_table_path=extracted_table_path)
    # available values are LinesRemovingState enums
    bbox_extraction_step_debug_path = debug_transformation_process(processor=bounding_box_extractor, img_handler=img_handler, state=BoundingBoxExtractionState.FINAL_BOUNDING_BOXES)
    print("Text Bounding Boxes Extraction Completed: \ndebug img: " + bbox_extraction_step_debug_path + "\n")

    # bounding boxes sorting
    bounding_box_sorter = TextBoundingSorter(
        bounding_boxes=bounding_boxes,
        table_columns=table_columns
    )
    table_bounding_box_array = bounding_box_sorter.run()

    # ocr extraction
    ocr_table = perform_ocr(
        bounding_box_array=table_bounding_box_array, 
        extracted_table_path=extracted_table_path,
        initial_img_path=image_path, 
        slices_folder=images_folder_path)
    print(ocr_table)

if __name__ == "__main__":
    main()