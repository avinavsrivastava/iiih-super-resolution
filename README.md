Test Edit

# iiih-super-resolution
Monsoon 2017: PGSSP Team 11

Steps for Executing SRCNN.py
1) Download SRCNN.py, train.zip, test.zip, and val.zip in local folder.
2) Extract train.zip, test.zip and val.zip in train, test and val folders respectively.
3) Use python 2.7.14
3) Install all dependent packages Keras, scipy, numpy, os, timeit, cv2.
4) Open terminal/command prompt and navigate to above download folder.
5) Execute command "python SRCNN.py"
6) Execution will take maximum 30 mins.
7) test folder will contain inuput low resolution images, bicubic images and predicted images.
8) Best PSNR achieved will be reported at the end of execution.

Steps for Executing SRKNN.py
1) Download SRKNN.py, train.zip and test.zip, in local folder.
2) Extract train.zip and test.zip in train, test folders respectively.
3) Use python 3.6.1
3) Install all dependent packages sklearn, scipy, numpy, os, matplotlib.
4) Open terminal/command prompt and navigate to above download folder.
5) Execute command "python SRKNN.py <Path of the train folder> <Path for the test folder> <Path for the output to be saved>"
6) Execution will take less than 5 mins.
7) Input high resolution, reconstructed image and blurred input images are saved in the output folder path given in the command
   line prompt.
8) PSNR will be reported on the console after every image reconstruction is completed in the test folder given in the command 
   line prompt.
