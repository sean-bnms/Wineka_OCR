<h1>Wineka: Data Collection with Optical Character Recognition</h1>

<h2>Objectives</h2>

This project contains the code used to perform Optical Character Recognition (OCR) to extract tabular data for the Wineka project. 

The initial code for the project is based on <a href="https://livefiredev.com/how-to-extract-table-from-image-in-python-opencv-ocr/">this article explaining how to apply OCR on an image to extract tabular data</a>. The <a href="https://github.com/livefiredev/ocr-extract-table-from-image-python"> public github repo <a/> can also be checked.

<h3>Data format</h3>
Here is a sample of the data we are trying to extract.

<h2>Optical Character Recognition: how does it work?</h2>
Optical Character Recognition is a technology used to extract text from images or scanned documents so it can be processed digitally. A common example is the conversion of a picture of a receipt into editable text.
<br/> <br/>
OCR rely on several steps, from preprocessing to character recognition: the goal of this section is to give an high level overview of the different concepts behind these steps. Hopefully, understanding these concepts should help framing OCR problems in the future and easily switch between libraries.

<h3>Preprocessing</h3>
This step is key: the cleanest the image used for the character recognition the better the quality of the output will be and it is good spending more time on these transformations to obtain good results.

<h4>Removing dependency to colors</h4>
OCR algorithms work more efficiently with images that are black and white for text extraction. Therefore the first step is to convert the image pixels to binary values, one for black, one for white (0 is usually applied to black pixels and 255 for white ones).
<br/> <br/>
It is usually done with two different operations:
<ul>
  <li><strong>grayscaling</strong>, which converts the image from a 3 dimensional space (e.g. RGB, where pixels are represented by their shade of red, green and blue like (123,200,255)) to a 1-dimension space (each pixel has a value from 0 to 255, 0 being black pixels and 255 being white pixels). This operation allows for faster processing of the image later on as we reduced dimensionality.</li>
  <li><strong>thresholding</strong>, which converts all shades of grey to either white or black value, based on the comparison between the pixel value (between 0 and 255) and the threshold value chosen. The method to calculate the threshold can vary based on the usecase (e.g. image where lightning vary), <a href="https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html">Open CV has an article describing well the different strategies that can be applied </a>.</li>
</ul>
Sometimes, depending on the transformations you want to apply, you might want to modify the binary color distribution of your previously preprocessed image. In this case, a useful transformation to apply is <strong>inverting</strong>, which converts black pixels to white pixels and respectively. For instance, in OpenCV the method to find contours looks for white objects on black background so inverting is often used as a preprocessing step.

<h4>Detecting contours in an image</h4>
Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. Detecting contours is a common operation when trying to detect objects. When trying to extract data from tables, we use contour detection at different steps:
<ul>
  <li>we want to detect the table external boundaries i.e. the main frame of the table</li>
  <li>we want to detect the table internal boundaries i.e. the columns and rows delimitations</li>
</ul>

To detect these contours we rely on two operations which results from <a href="https://dev.to/marcomoscatelli/a-gentle-introduction-to-convolutions-visually-explained-4c8d">convolution</a> operations:
<ul>
  <li>erosion</li>
  <li>dilation</li>
</ul>

It works like depicted below. For each contour of a table, you want first to select a kernel which follows the pattern of the contour (e.g. vertical line for columns delimiters, we will take a vertical kernel; horizontal kernel for an horizontal line like in the row delimiters). Then we will first apply erosion to remove any elements from the picture which is not following the contour pattern, then a dilation to amplify the contours and ensure detection.<a href="https://docs.opencv.org/4.x/dd/dd7/tutorial_morph_lines_detection.html"> OpenCV has also an article on selecting the kernel worth checking </a>.
<br/><br/>
![schema showcasing the difference between erosion and dilation with a horizontal kernel](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/morphologicalOperations.png?raw=true)

<h2>Running the project</h2>

<h3>Using the Streamlit App to review and clean the output data table from the OCR</h3>
It can be tedious to edit text which was not correctly recognized manually from your code editor. The goal of the Streamlit app created in the <strong>app.py</strong> file is to allow quick reviewing of the data obtained via the OCR and to quiclky clean the final .csv file obtained for optimal data quality.
<br/><br/>
![tab from the streamlit application allowing to troubleshoot the OCR process](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/app_2.png?raw=true)
