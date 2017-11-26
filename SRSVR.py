
from PIL import Image
from PIL import ImageFilter
import numpy as np
from sklearn.svm import SVR
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.feature_extraction import image
from skimage.measure import compare_psnr
import glob
import datetime
import ntpath
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random



    # Create Training data set
    # This function will create the train data set and return  X and Y

def createtraindata(path):
    X_R, Y_R,X_G,Y_G, X_B, Y_B=[],[],[],[],[],[]
   
    for ImageFile in glob.glob(path):
          Himage= Image.open(ImageFile)
          Limage= Himage.filter(ImageFilter.GaussianBlur(2))
          LimageArray=np.asarray(Limage,dtype="uint8")
        
          LimageDown=Limage.resize((int(Himage.width/2),int(Himage.height/2)),Image.ANTIALIAS)
                  
          #Interpolate the image to resize to original image dimension
          #LimageInterpolate = Image.fromarray(LimageArrayDown).resize((Himage.width,Himage.height),Image.BICUBIC);

          LimageInterpolate =LimageDown.resize((Himage.width,Himage.height),Image.BICUBIC);
                   
          #Get the R, G,B channel seperated
          LimageInterpolate_R=np.asarray(LimageInterpolate.getchannel(0))
          LimageInterpolate_G=np.asarray(LimageInterpolate.getchannel(1))
          LimageInterpolate_B=np.asarray(LimageInterpolate.getchannel(2))
                    
          #Get the corresponding H Channels
          Himage_R=np.asarray(Himage.getchannel(0))
          Himage_G=np.asarray(Himage.getchannel(1))
          Himage_B=np.asarray(Himage.getchannel(2))
     
                 
          #Generate the training set for each channel
          # Here we have used non-overlaping step size. It can be 1 and it will take more time
          stepSize=5      
          #for R : Patch Size 5X5
          Patch_shape = (5, 5)    
                            
          LPatches_R= extract_patches_2d(LimageInterpolate_R,Patch_shape)
          HPatches_R= extract_patches_2d(Himage_R,Patch_shape)
          # flatten the patch
          x_r=LPatches_R.reshape(LPatches_R.shape[0],LPatches_R.shape[1]*LPatches_R.shape[2])
          y_r=HPatches_R[:,int(HPatches_R.shape[1]/2),int(HPatches_R.shape[2]/2)]
          #Push this row to the traning set
          X_R.append(x_r)
          Y_R.append(y_r)
                   
          LPatches_G= extract_patches_2d(LimageInterpolate_G,Patch_shape)
          HPatches_G= extract_patches_2d(Himage_G,Patch_shape)
          # flatten the patch
          x_g=LPatches_G.reshape(LPatches_G.shape[0],LPatches_G.shape[1]*LPatches_G.shape[2])
          y_g=HPatches_G[:,int(HPatches_G.shape[1]/2),int(HPatches_G.shape[2]/2)]
          #Push this row to the traning set
          X_G.append(x_g)
          Y_G.append(y_g)
                     
          LPatches_B= extract_patches_2d(LimageInterpolate_B,Patch_shape)
          HPatches_B= extract_patches_2d(Himage_B,Patch_shape)
          # flatten the patch
          x_b=LPatches_B.reshape(LPatches_B.shape[0],LPatches_B.shape[1]*LPatches_B.shape[2])
          y_b=HPatches_B[:,int(HPatches_B.shape[1]/2),int(HPatches_B.shape[2]/2)]
          #Push this row to the traning set
          X_B.append(x_b)
          Y_B.append(y_b)
         
    return X_R,Y_R,X_G,Y_G,X_B,Y_B

def CreateTestData(ImageFile):
          outDirName="Result"
          XP_R, XP_G, XP_B=[],[],[]
                  
          Himage= Image.open(ImageFile)
          #Limage= Himage.filter(ImageFilter.GaussianBlur(2))
          Limage=Himage
          LimageArray=np.asarray(Limage,dtype="uint8")
                 
          LimageDown=Limage.resize((int(Himage.width/2),int(Himage.height/2)),Image.BICUBIC)
          LimageInterpolate =LimageDown.resize((Himage.width,Himage.height),Image.BICUBIC);

          #save the low dimension pic
          print('Save low resolution picture of ' + str(ImageFile))
          filename=ntpath.basename(ImageFile)
          LimageInterpolate.save(outDirName +'//' + 'LR_' + str(filename))
                   
          #Get the R, G,B channel seperated
          LimageInterpolate_R=np.asarray(LimageInterpolate.getchannel(0))
          LimageInterpolate_G=np.asarray(LimageInterpolate.getchannel(1))
          LimageInterpolate_B=np.asarray(LimageInterpolate.getchannel(2))
                          
          #Generate the training set for each channel
          # Here we have used non-overlaping step size. It can be 1 and it will take more time

          stepSize=5    
          #for R : Patch Size 5X5
          Patch_shape = (5, 5)    
                               
          LPatches_R= extract_patches_2d(LimageInterpolate_R,Patch_shape)
          LPatches_G= extract_patches_2d(LimageInterpolate_G,Patch_shape)
          LPatches_B= extract_patches_2d(LimageInterpolate_B,Patch_shape)

          #HPatches_R= extract_patches_2d(Himage_R,Patch_shape)
          # flatten the patch
          x_r=LPatches_R.reshape(LPatches_R.shape[0],LPatches_R.shape[1]*LPatches_R.shape[2])
          x_g=LPatches_G.reshape(LPatches_G.shape[0],LPatches_G.shape[1]*LPatches_G.shape[2])
          x_b=LPatches_B.reshape(LPatches_B.shape[0],LPatches_B.shape[1]*LPatches_B.shape[2])

          #Push this row to the Image construction set
          XP_R.append(x_r)
          XP_G.append(x_g)
          XP_B.append(x_b)
    
          return XP_R,XP_G,XP_B


def main():

    Trainfolder = "Train/*.jpg"
    Testfolder = "Test/*.jpg"
    outDirName="Result"

    samplesize_K=10000

    PSNR_Mean =[]
      # retrieve 2p+1 X 2p+1 patch from Lower resolution image and create X, Select the corresponding center pixel from H image
        # and create y
        # Generate the R, G, B component 
        # Create the feature set for each channel ( R,G,B)
        # Create the Y value for each channel for each channel from Original Image
    print('Started Creating training Data @ ' + str(datetime.datetime.now().time()))          
    XT_R,YT_R,XT_G,YT_G,XT_B,YT_B=createtraindata(Trainfolder)
    print('Completed training Data')     
    # Call the Fit function for SVR for each channel
    # #############################################################################
    # Fit regression model
    svr_R = SVR(kernel='rbf', C=362, epsilon=0.2)
    XR = np.array(XT_R)[0,:,:]
    YR= np.array(YT_R)[0,:]
    
    #XR = preprocessing.scale(XR)
    #YR = preprocessing.scale(YR)

    idx = np.random.choice(np.arange(XR.shape[0]), samplesize_K, replace=False)
    XR=XR[idx,:]
    YR=YR[idx]
    XR=XR * (1/255)
    YR=YR * (1/255)

    print('Training Start for R: ' + str(datetime.datetime.now().time()))
    Model_R = svr_R.fit(XR,YR)
    print('Training End for R: ' + str(datetime.datetime.now().time()))

   
    svr_G = SVR(kernel='rbf', C=362, epsilon=0.2)
    XG = np.array(XT_G)[0,:,:]
    YG= np.array(YT_G)[0,:]
    #XG = preprocessing.scale(XG)
    #YG = preprocessing.scale(YG)

   
    XG=XG[idx,:]
    YG=YG[idx]
    XG=XG * (1/255)
    YG=YG * (1/255)

    
    print('Training Start for G: ' + str(datetime.datetime.now().time()))
    Model_G = svr_G.fit(XG,YG)
    print('Training End for G: ' + str(datetime.datetime.now().time()))
    
    svr_B = SVR(kernel='rbf', C=362, epsilon=0.2)
    XB = np.array(XT_B)[0,:,:]
    YB= np.array(YT_B)[0,:]
    #XB = preprocessing.scale(XB)
    #YB = preprocessing.scale(YB)

    XB=XB[idx,:]
    YB=YB[idx]
    XB=XB * (1/255)
    YB=YB * (1/255)

    print('Training Start for B: ' + str(datetime.datetime.now().time()))
    Model_B = svr_B.fit(XB,YB)
    print('Training End for B: ' + str(datetime.datetime.now().time()))

    
    print('Started Testing Data')  
    # Use the Test data for predict
    for ImageFile in glob.glob(Testfolder):
        Himage= Image.open(ImageFile)
        print('Started Testing for: ' + str(ImageFile) + ' @ ' +  str(datetime.datetime.now().time()))  
        XP_R,XP_G,XP_B=CreateTestData(ImageFile)
        print('Created  Test data for : ' + str(ImageFile) + ' @ ' +  str(datetime.datetime.now().time()))  

        #XP_R=preprocessing.scale(np.array(XP_R)[0,:,:])
        #XP_G=preprocessing.scale(np.array(XP_G)[0,:,:])
        #XP_B=preprocessing.scale(np.array(XP_B)[0,:,:])

        XP_R=np.array(XP_R)[0,:,:] * (1/255)
        XP_G=np.array(XP_G)[0,:,:] * (1/255)
        XP_B=np.array(XP_B)[0,:,:] * (1/255)
                      
        print('Predicting data for R : ' + ' @ ' +  str(datetime.datetime.now().time())) 
        YP_R=Model_R.predict(XP_R)
        print('Prediction Complete for R : ' + ' @ ' +  str(datetime.datetime.now().time())) 

        print('Predicting data for G : ' + ' @ ' +  str(datetime.datetime.now().time())) 
        YP_G=Model_G.predict(XP_G)
        print('Prediction Complete for G : ' + ' @ ' +  str(datetime.datetime.now().time())) 


        print('Predicting data for B : ' + ' @ ' +  str(datetime.datetime.now().time())) 
        YP_B=Model_B.predict(XP_B)
        print('Prediction Complete for B : ' + ' @ ' +  str(datetime.datetime.now().time())) 
               
        ImageShape=(Himage.height,Himage.width,3)
        Patch_shape = (5, 5)   
        
        NewImageArray=np.zeros(ImageShape,dtype=np.uint8)
        NewImage_R=NewImageArray[:,:,0]

        print('Creating patch data for R : ' + ' @ ' +  str(datetime.datetime.now().time())) 
        Patches_R= extract_patches_2d(NewImage_R,Patch_shape)
        i=0
        for patch in Patches_R:
         patch[:,:]=int(YP_R[i] * 255)
         i=i+1
        
        print('Reconstructing R image from patches: ' + ' @ ' +  str(datetime.datetime.now().time())) 
        Reconstruct_R= reconstruct_from_patches_2d(Patches_R,(Himage.height,Himage.width))
        
        print('Creating patch data for G : ' + ' @ ' +  str(datetime.datetime.now().time())) 
        NewImage_G=NewImageArray[:,:,1]

        Patches_G= extract_patches_2d(NewImage_G,Patch_shape)
        i=0
        for patch in Patches_G:
         patch[:,:]=int(YP_G[i] * 255)
         i=i+1
        
        print('Reconstructing G image from patches: ' + ' @ ' +  str(datetime.datetime.now().time()))
        Reconstruct_G= reconstruct_from_patches_2d(Patches_G,(Himage.height,Himage.width))

        print('Creating patch data for B : ' + ' @ ' +  str(datetime.datetime.now().time())) 
        NewImage_B=NewImageArray[:,:,2]

        
        Patches_B= extract_patches_2d(NewImage_B,Patch_shape)
        i=0
        for patch in Patches_B:
         patch[:,:]=int(YP_B[i] * 255)
         i=i+1
                 
        
        print('Reconstructing B image from patches: ' + ' @ ' +  str(datetime.datetime.now().time()))
        Reconstruct_B= reconstruct_from_patches_2d(Patches_B,(Himage.height,Himage.width))
        
        print('Started image contstructing for: ' + str(ImageFile) + ' @ ' +  str(datetime.datetime.now().time()))  
        reconstruct_RGB = np.dstack((Reconstruct_R,Reconstruct_G,Reconstruct_B))
        reconstruct_RGB= reconstruct_RGB.astype(np.uint8)

        print('Completed image contstructing for: ' + str(ImageFile) + ' @ ' +  str(datetime.datetime.now().time()))  
        HimageA= np.asarray(Himage,dtype=np.uint8)
        
        print('Calculating PSNR for: ' + str(ImageFile) + ' @ ' +  str(datetime.datetime.now().time()))  
         # Create the PSNR value 
        PSNR=compare_psnr(HimageA,reconstruct_RGB)
        print ('PSNR calculated for : ' + str(ImageFile) + ' : ' + str(PSNR))
        
         #save the reconstructed dimension pic
        print('Save reconstructed picture of ' + str(ImageFile))
        filename=ntpath.basename(ImageFile)
        Image.fromarray(reconstruct_RGB).save(outDirName +'//' + 'SR_' + str(filename))
        PSNR_Mean.append(PSNR)

    print('Image Super resolution completed')
    print('PSNR values obtained for the reconstruction :' + str(PSNR_Mean))
    print('Mean PSNR :'+ str(np.mean(PSNR_Mean)))
   

        
                     
    # Ask for Prediction function

                
     
     
if __name__=="__main__":
    main()
 








