from dataclasses import dataclass
from typing import Protocol

# allows modules to access modules from outside the package
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

# import modules from the project
import image_processing as image_processing


class Thresholder(Protocol):
    def apply_threshold(self, image:image_processing.Image) -> image_processing.Image:
        ...

@dataclass
class GlobalThresholder:
    new_value: int = 255
    inversion: bool = False
    static_threshold_value: float = 127

    def apply_threshold(self, image:image_processing.Image):
        return image_processing.apply_simple_binary_threshold(image=image, threshold_value=self.static_threshold_value, new_pixel_value=self.new_value,inversion=self.inversion)


@dataclass
class GlobalOptimizedThresholder:
    new_value: int = 255
    inversion: bool = False

    def apply_threshold(self, image:image_processing.Image):
        return image_processing.apply_complex_binary_threshold(image=image, new_pixel_value=self.new_value, inversion=self.inversion)

@dataclass
class AdaptiveThresholder:
    method: image_processing.AdaptiveThresholdMethod
    new_value: float = 255
    inversion: bool = False

    def apply_threshold(self, image:image_processing.Image):
        return image_processing.apply_adaptive_threshold(image=image, method=self.method, new_pixel_value=self.new_value, inversion=self.inversion)



@dataclass
class ImagePreProcessor:
    '''
    Turn a colored image into a black and white binary representation.
    - image: the mathematical representation of the source image
    - thresholder: the type of threshold used to obtain a binary representation of the image
    '''
    image: image_processing.Image
    thresholder: Thresholder

    def apply(self) -> image_processing.Image:
        self.grayscaled_image = image_processing.convert_image_to_grayscale(image=self.image)
        self.thresholded_image = self.thresholder.apply_threshold(image=self.grayscaled_image)
        return self.thresholded_image