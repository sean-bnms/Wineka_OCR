import image_processing
from dataclasses import dataclass, field

from MorphologicalTransformer import MorphologicalTransformer, MorphologicalOperation


@dataclass
class Color:
    '''
    Allows to represent a color in different color modes. Currently supported are RGB and HSV modes.
    - rgb_color: contains the value of red, blue and green from the RGB color representation, each with a value between 0 and 255
    '''
    rgb_color: tuple[int,int,int]

    # HSV color model explanation can be found here: https://en.wikipedia.org/wiki/HSL_and_HSV 
    def get_hsv_color(self) -> tuple[float,float,float]: 
        '''
        Converts RGB (Red, Green, Blue) color provided to the HSV (Hue, Saturation, Value) color mode.
        '''
        if not (0<self.rgb_color[0]<255 and 0<self.rgb_color[1]<255 and 0<self.rgb_color[2]<255):
            raise ValueError("R,G,B values should be between 0 and 255")

        r, g, b = self.rgb_color[0] / 255.0, self.rgb_color[1] / 255.0, self.rgb_color[2] / 255.0  # Normalize to [0, 1]
        
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val  #also called the chroma

        # Hue
        if delta == 0:
            h = 0
        elif max_val == r:
            h = 60 * (((g - b) / delta) % 6)
        elif max_val == g:
            h = 60 * (((b - r) / delta) + 2)
        else:  # max_val == b
            h = 60 * (((r - g) / delta) + 4)
        
        # Saturation
        s = 0 if max_val == 0 else delta / max_val
        # Value
        v = max_val
        return round(h, 2), round(s, 2), round(v, 2)

    # OpenCV implementes HSV in a specific manner for optimization purposes
    def get_opencv_hsv_color(self) -> tuple[int, int, int]:
        h, s, v = self.get_hsv_color()
        h_opencv = h / 2
        s_opencv = s * 255
        v_opencv = v * 255
        return round(h_opencv), round(s_opencv), round(v_opencv)
    
    def set_hue(self) -> None:
        self.h = self.get_opencv_hsv_color()[0]
    
    # boundaries calculation methods are explained here: https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html 
    def get_hsv_boundaries(self, tolerance_h: int) -> list[tuple[list[int], list[int]]]:
        '''
        Computes the HSV boundaries to apply for a given HSV color, computed with OpenCV definition of HSV.
        - h: value of hue from the HSV color, between 0 and 180
        - tolerance_h: margin around a hue to account to variations due to original image quality, defaulted to 10
        '''
        # because of hue overlaps (360 degrees values), we include edge conditions to cover the
        # h = 8, h-10 = -2 -> [178,180], h+10 = 18 -> [0,18]
        # h = 172, h-10 = 162 -> [162, 180], h+10 = 182 -> [0,2]
        self.set_hue()
        boundaries = []
        if self.h < tolerance_h:
            boundaries1 = [self.h - tolerance_h + 180, 20, 20], [180, 255, 255]
            boundaries2 = [0, 20, 20], [self.h + tolerance_h, 255, 255]
            boundaries.append(boundaries1)
            boundaries.append(boundaries2)
        elif self.h > 180 - tolerance_h:
            boundaries1 = [self.h - tolerance_h, 20, 20], [180, 255, 255]
            boundaries2 = [0, 20, 20], [(self.h + tolerance_h) % 180, 255, 255]
            boundaries.append(boundaries1)
            boundaries.append(boundaries2)
        else:     
            lower = [self.h - tolerance_h, 20, 20]
            upper = [self.h + tolerance_h, 255, 255]
            boundaries.append((lower, upper))
        return boundaries



@dataclass
class ColorFilter:
    '''
    Handles the filtering of a RGB color from an image.
    - image: the image outcome from the OpenCV .imread method
    - color: the Color object corresponding to the color we want to filter 
    - hue_tolerance: the tolerance used to create the HSV color boundaries for filtering, results in [h-10,...] [h+10, ...] intervals
    - kernel: the Kernel object we want to apply for filtering, defaulted to a 20,20 ellipse
    '''
    color: Color
    image: image_processing.Image
    hue_tolerance: int = 10
    kernel: image_processing.Kernel = field(default_factory=lambda: image_processing.Kernel(shape=image_processing.KernelShape.ELLIPSE, dimensions=(20,20)))

    def create_color_mask(self, hsv: image_processing.Image):
        '''
        Creates for a given image and a RGB (Red, Green, Blue) color the mask to apply on the image to filter out the color.
        - hsv: image we want to filter color from, represented via the HSV channels computed in OpenCV 
        '''
        color_boundaries = self.color.get_hsv_boundaries(tolerance_h=self.hue_tolerance)
        # mask initialization
        mask = image_processing.create_mask(image=hsv, boundaries=(color_boundaries[0][0], color_boundaries[0][1]))
        nbr_boundaries = len(color_boundaries)
        if nbr_boundaries > 1:
            for i in range(1, nbr_boundaries):
                mask += image_processing.create_mask(image=hsv, boundaries=(color_boundaries[i][0], color_boundaries[i][1]))
        return mask
    
    def filter(self) -> image_processing.Image:
        '''
        Turns all pixels from the image representing a color within the range provided to black.
        '''
        # leverages HSV color mode representation for more accurate filtering
        hsv = image_processing.convert_image_to_hsv(image=self.image)
        mask = self.create_color_mask(hsv=hsv) 
            
        # apply a morphology operation, closing: it results in a Dilation followed by Erosion
        # helps filling small pixel gaps in an area (e.g. holes in a shape because of different lighting conditions)
        morph_transformer = MorphologicalTransformer(image=mask, operation=MorphologicalOperation.CLOSING, kernel=self.kernel)
        morph = morph_transformer.apply()

        mask = 255 - morph
        filtered_image = image_processing.apply_bitwise_and(image1=self.image, image2=self.image, mask=mask)
        return filtered_image

    
    
    