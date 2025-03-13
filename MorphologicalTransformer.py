from dataclasses import dataclass, field
from enum import StrEnum, auto

import image_processing

class MorphologicalOperation(StrEnum):
    DILATION = auto()
    EROSION = auto()
    OPENING = auto()
    CLOSING = auto()

@dataclass
class MorphologicalTransformer:
    image: image_processing.Image
    operation: MorphologicalOperation
    kernel: image_processing.Kernel =  field(
        default_factory= lambda: image_processing.Kernel(shape=image_processing.KernelShape.RECTANGLE, dimensions=(3,3)))
    nbr_iterations: int = 1

    def apply(self) -> image_processing.Image:
        match self.operation:
            case MorphologicalOperation.DILATION:
                transformed_image = image_processing.dilate(image=self.image, kernel=self.kernel, nbr_iterations=self.nbr_iterations)
            case MorphologicalOperation.EROSION:
                transformed_image = image_processing.erode(image=self.image, kernel=self.kernel, nbr_iterations=self.nbr_iterations)
            case MorphologicalOperation.OPENING:
                transformed_image = image_processing.open(image=self.image, kernel=self.kernel, nbr_iterations=self.nbr_iterations)
            case MorphologicalOperation.CLOSING:
                transformed_image = image_processing.close(image=self.image, kernel=self.kernel, nbr_iterations=self.nbr_iterations)
        return transformed_image
