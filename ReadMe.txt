############################################################################
#
#  Math 545 Final Project
#
#  Professor: Qian-Yong Chen
#  Students: Yi Fung, William He
#
#  Due Date: April 30, 2018
#
#
#  Thank you for grading and have a great summer!
#
############################################################################


This is the ReadMe file providing some documentations explaining how to  
navigate through and make use of our repository.


Introduction:

Our project topic is the application of singular valude decomposition in image 
compression. 

Specifically, we collected a dataset of 20 100x100 images of famous people
portraits. We then performed SVD on the 20-by-10000 input data and used principle 
component analysis (a specific usage of SVD) to find the eigenvectors and compress
the data.

We used compression rate and reconstruction error as our evaluation metrics in
exploring how different number of components used effects the quality of image
compression.


Notes about the code environment:

- We wrote our code in python3.
  To run a python file, you open the terminal and go into the directory of the project folder.
  Then, type in the terminal >> python3 file_name.py
  for whatever file you want to run.
  (Need to install python first if that's not on your machine).

- You may need to install some of the library and package dependences in order to successfully run the python file.
  To do that, type in the terminal >> sudo pip install library_name
  where library_name would be something like numpy (the library to deal with arrays in python).


Notes about the structure of the code and files in this directory:

- driver.py is the python file that reformats image scraped from the web into 100x100 images,
  in either greyscale or colored format. Run this first.

- run_me.py is the python file that runs SVD / PCA on image data, performs the compression, and reconstructs 
  the image data. Compression rate, reconstruction error, and plots are generated here.

- imgutils.py is the python file that contains some helper function to deal with image analysis. 
  Called by the main python files.

- The pics folder contains the black and white images used.
  The pics2 folder contains the colored images used.

- The figures folder contains the plots generated showing the PCA compression and reconstruction results.



If there are any additional questions regarding how to navigate and run the code, 
feel to contact Yi (yfung@umass.edu) or William (ziweihe@umass.edu). Thanks.
