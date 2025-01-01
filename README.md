<h1>Wineka: Data Collection with Optical Character Recognition</h1>

<h2>Objectives</h2>

This project contains the code used to perform Optical Character Recognition (OCR) to extract tabular data for the Wineka project. 

The initial code for the project is based on <a href="https://livefiredev.com/how-to-extract-table-from-image-in-python-opencv-ocr/">this article explaining how to apply OCR on an image to extract tabular data</a>. The <a href="https://github.com/livefiredev/ocr-extract-table-from-image-python"> public github repo <a/> can also be checked.

<h3>Data format</h3>
Here is a sample of the data we are trying to extract.

<h2>Optical Character Recognition: how does it work?</h2>
Optical Character Recognition is a technology used to extract text from images or scanned documents so it can be processed digitally. A common example is the conversion of a picture of a receipt into editable text.
<br/>
<br/>
OCR rely on several steps, from preprocessing to character recognition: the goal of this section is to give an high level overview of the different concepts behind these steps. Hopefully, understanding these concepts should help framing OCR problems in the future and easily switch between libraries.

<h3>Preprocessing</h3>
This step is key: the cleanest the image used for the character recognition the better the quality of the output will be and it is good spending more time on these transformations to obtain good results.

<h2>Running the project</h2>

<h3>Using the Streamlit App to review and clean the output data table from the OCR</h3>
It can be tedious to edit text which was not correctly recognized manually from your code editor. The goal of the Streamlit app created in the <strong>app.py</strong> file is to allow quick reviewing of the data obtained via the OCR and to quiclky clean the final .csv file obtained for optimal data quality.

![tab from the streamlit application allowing to troubleshoot the OCR process](https://github.com/sean-bnms/Wineka_OCR/blob/main/resources/app_2.png?raw=true)
