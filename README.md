<h1>Wineka: Data Collection with Optical Character Recognition</h1>

<h2>Objectives</h2>

This project contains the code used to perform Optical Character Recognition (OCR) to extract tabular data for the Wineka project. 

The initial code for the project is based on <a href="https://livefiredev.com/how-to-extract-table-from-image-in-python-opencv-ocr/">this article explaining how to apply OCR on an image to extract tabular data</a>. The <a href="https://github.com/livefiredev/ocr-extract-table-from-image-python"> public github repo <a/> can also be checked.

<h3>Data format</h3>
Here is a sample of the data we are trying to extract. It contains information about wine pairings with meals. It presents a twist compared to usual table extraction use cases as the tables do not only contain text but also colored icons which adds erosion steps during preprocessing.
<br/><br/>

![sample of the tables we want to extract content from](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/IMG_0147.jpg?raw=true)


<h2>Optical Character Recognition on tabular data: how does it work?</h2>
Optical Character Recognition is a technology used to extract text from images or scanned documents so it can be processed digitally. A common example is the conversion of a picture of a receipt into editable text.
<br/> <br/>
OCR rely on several steps, from preprocessing to character recognition: the goal of this section is to give an high level overview of the different concepts behind these steps. Hopefully, understanding these concepts should help framing OCR problems in the future and easily switch between libraries.

<h3>Preprocessing: isolating the text from the table</h3>
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
  <li><strong>erosion</strong></li>
  <li><strong>dilation</strong></li>
</ul>

It works like depicted below. For each contour of a table, you want first to select a kernel which follows the pattern of the contour (e.g. vertical line for columns delimiters, we will take a vertical kernel; horizontal kernel for an horizontal line like in the row delimiters). Then we will first apply erosion to remove any elements from the picture which is not following the contour pattern, then a dilation to amplify the contours and ensure detection.<a href="https://docs.opencv.org/4.x/dd/dd7/tutorial_morph_lines_detection.html"> OpenCV has also an article on selecting the kernel worth checking </a>.
<br/><br/>

![schema showcasing the difference between erosion and dilation with a horizontal kernel](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/morphologicalOperations.png?raw=true)

<h4>Removing contours from an image</h4>
Once we have isolated the contours we are trying to remove, we can rely on pixel <strong>addition</strong> and <strong>substraction</strong> operations. It consists in combining the pixels of two images by leveraging the corresponding operations on matrixes; we usually:
<ul>
  <li>add all the eroded images of the table lines togehter, before dilating the resulting image</li>
  <li>then substract this image with all the table lines we want to remove to the original inverted image, which results in the image without the table lines</li>
</ul>

<h3>Text Extraction: gathering the content of the different cells and translating images to text</h3>

<h4>Detecting the text from each cells</h4>
This step is using the same fundamentals as described earlier. We:
<ul>
  <li>dilate the text blocks with a kernel allowing to discard the spaces between words / letters. The goal is to have a dilation strong enough to create blocks of pixels uniformly distributed (here, all will have a value of 1 or 255 as they are white): these blocks are called <strong>blobs</strong></li>
  <li>then perform blob detection, which means we find the contour boxes of each blob, which will help find the contours of the text</li>
  <li>then store each of the text boxes in an array to retain the order of the table to be able to reconstruct it later </li>
  <li>and finally, slice the image to obtain, for each text box, an image containing only its text</li>
</ul>

<h4>Extracting the text from each sliced image</h4>
Finally, we use an OCR model to extract each piece of text from the sliced images: different services exist, for this project the open source models from Tesseract were sufficient (<a href="https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/">this article is worth checking to learn how to tweak Tesseract parameters</a>). 
Once the text outputs are all collected, you can store them in a .csv or a database table by reordering the text boxes based on the array created at the previous step.

<h2>Understanding the code</h2>

<h3>Extracting the table from the initial image</h3>
The first step that the code realizes is to extract the full table from the image. The logic is implemented within the TableExtractor class.
<br/><br/>
First we preprocess the data as showcased below.

![visual representation of the data transformation occuring on the data sample for table extraction preprocessing](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/tableExtraction_1.png?raw=true)

Then we find all contours in the image and, based on the corner coordonates of the biggest contour, and crop the image accordingly (a transformation is then applied to correct the perspective of the extracted table and add padding around it to make next transformations easier).

![visual representation of the data transformation occuring on the data sample for table extraction based on contours](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/tableExtraction_2.png?raw=true)

<h3>Removing the lines of the table and the icons</h3>
Then we want to remove the table structure as well as the icons so we only have the text remaining. The logic is implemented in the TableLinesAndIconsRemover class.
<br/><br/>
We apply different kernels to be able to erode the vertical and horizontal lines of the table as well as the icons it contains. After that, we add the different eroded images, dilate the resulting image and substract it to the inverted image of the table extracted. A cleaning transformation is then applied to remove some of the remaining noise on the image (e.g. lines not completely eroroded because of the image deformations induced by the book structure, by the lightning when the picture was taken...).

![visual representation of the data transformation occuring on the data sample for table extraction based on contours](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/tableErosion.png?raw=true)

<h3>Detecting the text and slicing the image accordingly</h3>
We then want to extract each piece of text contained in the cells, apply OCR on each of them and recreate the table in a .csv file. The logic for this is implemented in the OcrToTableTool class.
<br/><br/>
First we detect the blobs from the image obtained after table and icons erosion.

![visual representation of the data transformation occuring on the data sample for blob detection](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/blobDetection.png?raw=true)

Then we identify the correct text blobs from all blobs collected: a custom logic is implemented to remove non sought blobs. We keep track of the order of the different text boxes to be able to assign each text box to the correct cell when recreating the table later. The different text boxes are then extracted from the image as separated slices: we later apply an OCR model on them to obtain the extracted data.

![visual representation of the data transformation occuring on the data sample for text slicing](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/textSlicing.png?raw=true)

Finally, we recreate the table in a .csv file. A Streamlit app can then be used to easily analyze the data obtained and quickly make edits if needed.

<h2>Running the project</h2>

<h3>Using the Streamlit App to review and clean the output data table from the OCR</h3>
It can be tedious to edit text which was not correctly recognized with the OCR model manually from your code editor. The goal of the Streamlit app created in the <strong>app.py</strong> file is to allow quick reviewing of the data obtained via the OCR and to quiclky clean the final .csv file obtained for optimal data quality.
<br/><br/>

![tab from the streamlit application allowing to troubleshoot the OCR process](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/app_2.png?raw=true)
