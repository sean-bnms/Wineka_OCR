from typing import Protocol
from enum import StrEnum

from cv_operations.ImagePreProcessor import GlobalOptimizedThresholder, GlobalThresholder
from ocr_table_operations.TableExtractor import TableExtractor, TableExtractionState
from ocr_table_operations.TableIconsRemover import TableIconsRemover, IconRemovingState
from ocr_table_operations.TableLinesRemover import TableLinesRemover, LinesRemovingState
from ocr_table_operations.TextBoundingBoxExtractor import TextBoundingBoxExtractor, BoundingBox, BoundingBoxExtractionState
from ocr_table_operations.TextBoundingSorter import TextBoundingSorter
from ocr_table_operations.OcrProcessor import OrcProcessor, TesseractLanguage
from image_processing import ImageHandler, Kernel, KernelShape, Image
import ocr_result_processing

DEBUG_FOLDER_PATH = "images/debug/"
OCR_SLICES_FOLDER_PATH = "images/ocr_slices/"
COLUMN_NAMES = ["Plat", "Type de vin", "Appelation"]

def extract_table(background_color: tuple[int,int,int], image: Image) -> tuple[TableExtractor,Image]:
    # otsu optimized thresholding method showed better results
    # to perform table extraction
    table_extractor = TableExtractor(
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

def perform_ocr(bounding_box_array:list, extracted_table_path:str, initial_img_path:str, slices_folder:str) -> list[list[str]]:
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

### Execute
def retrieve_csv(image_path:str, background_color:str, icons_colors:list[tuple[int,int,int]]) -> None:
    # initialize the image handler
    img_handler = ImageHandler(image_path=image_path)
    image = img_handler.load_image()

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
    icon_remover, image_without_icons = remove_icons(icon_colors=icons_colors, image_path=extracted_table_path)
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
        slices_folder=OCR_SLICES_FOLDER_PATH)
    print("OCR Completed: \n")
    print(ocr_table)
    
    # ocr saving as csv
    # cleans first the bullet point incorrectly recognized
    ocr_cleaned_table = ocr_result_processing.clean_bullet_points(raw_table=ocr_table)
    csv_path = ocr_result_processing.store_table_as_csv(table=ocr_cleaned_table, column_names=COLUMN_NAMES, csv_name=img_handler.get_image_name())
    print("CSV Table Saved: \n" + csv_path + "\n")






def main():
    # some variables for the table extraction
    background_color = (163, 151, 152)
    gold_icons_color = (158, 130, 90)
    red_icons_color = (163, 151, 152)
    icons_colors = [gold_icons_color, red_icons_color]

    # path to store and load images
    image_path = "images/IMG_0151.jpg"
    retrieve_csv(
        image_path=image_path,
        background_color=background_color,
        icons_colors=icons_colors
        )

    # for i in range(1):
    #     # path to store and load images
    #     image_path = f"images/IMG_0{147+i}.jpg"
    #     retrieve_csv(
    #         image_path=image_path,
    #         background_color=background_color,
    #         icons_colors=icons_colors
    #         )

if __name__ == "__main__":
    main()