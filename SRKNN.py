from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.feature_extraction import image
import numpy as np
from skimage.util.shape import view_as_blocks
from skimage.measure import compare_psnr
import os,sys

H_patch=6
L_patch=3
KNN=3
type="double"
channels=3

def make_data(image):
	patches_H=view_as_blocks(image,(H_patch,H_patch))
	image_blur=ndimage.gaussian_filter(image,sigma=3)

	image_down=np.zeros(shape=(int(image.shape[0]/2),int(image.shape[1]/2)))


	image_down=misc.imresize(image_blur,((int(image.shape[0]/2),int(image.shape[1]/2))))
	patches_blur=view_as_blocks(image_down,(L_patch,L_patch))

	i,j,k=0,0,0
	train_data = np.zeros(shape=(patches_blur.shape[0]*patches_blur.shape[1],H_patch))

	while i < patches_blur.shape[0]:
		j = 0
		while j < patches_blur.shape[1]:
			train_data[k] = np.resize(patches_blur[i][j],(1,H_patch)).astype(type)
			k=k+1
			j=j+1
		i=i+1

	image_inter=misc.imresize(image_down,(int(image.shape[0]),int(image.shape[1])),"bicubic")
	#image_inter=ndimage.zoom(image_down,2,order=3)

	patches_lup=view_as_blocks(image_inter,(H_patch,H_patch))
	
	
	diff=patches_H.astype(type)-patches_lup.astype(type)

	train_label = np.zeros(shape=(diff.shape[0]*diff.shape[1],H_patch*H_patch))
	i,j,k=0,0,0
	while i < diff.shape[0]:
		j = 0
		while j < diff.shape[1]:
			train_label[k] = np.resize(diff[i][j],(1,H_patch*H_patch)).astype(type)
			k=k+1
			j=j+1
		i=i+1
	return train_data, train_label
	
def get_train_data(path):
	first = 0

	knn_train_data=np.zeros(shape=(100,4))
	knn_train_label=np.zeros(shape=(100,16))
	
	for x in os.listdir(path):
		fname = os.path.join(path, x)
		print("fname -\n",x,first)
		if(fname.endswith(".jpg")):
			height,width,channel = ndimage.imread(fname).shape
	
			im=misc.imread(fname,mode='RGB')
			
			if(im.shape[0] % H_patch != 0):
				width = width - (width%H_patch)
			if(im.shape[1] % H_patch != 0):
				height = height - (height%H_patch)
			im = misc.imresize(im,(height,width))

			count = 0
			while count < channels:

				image = im[:,:,count]

				if (first == 0):
					knn_train_data, knn_train_label=make_data(image)
					
				else:
					temp_data,temp_label=make_data(image)
					knn_train_data=np.vstack([knn_train_data,temp_data])
					knn_train_label=np.vstack([knn_train_label,temp_label])

				count = count + 1
		first = first + 1		
	return knn_train_data, knn_train_label


if __name__ == '__main__':

	# Check for the correct number of arguments
	if len(sys.argv) != 4:
		print("number of required arguments is not correct\n")
		print("Correct format is : python srknn.py <train directory path> <test directory path> <output file save directory path>")
		exit(0)
		
	path1=sys.argv[1]
	path=sys.argv[2]
	output_path=sys.argv[3]
	#Derive the model data required for KNN from the images
	knn_train_data,knn_train_label=get_train_data(path1)
	
	# Train KNN model
	clf=neighbors.KNeighborsClassifier(n_neighbors=KNN)
	clf.fit(knn_train_data,knn_train_label)
	
	print("After training the model")
	
	# Go through the test directory for the images to predict
	for x in os.listdir(path):
		fname = os.path.join(path, x)
		print("fname\n",x)
		if(fname.endswith(".jpg")):
	
			#Reconstruction of image
			im=misc.imread(fname,mode='RGB')
			height=im.shape[1]
			width=im.shape[0]
			if(im.shape[0] % H_patch != 0):
				width = width - (width%H_patch)
			if(im.shape[1] % H_patch != 0):
				height = height - (height%H_patch)
			
			#Resize the image for the given High resolution window patch
			im = misc.imresize(im,(width,height))

			# To capture the reconstructed image
			blue_output=np.zeros(shape=(im.shape[0],im.shape[1]))
			green_output=np.zeros(shape=(im.shape[0],im.shape[1]))
			red_output=np.zeros(shape=(im.shape[0],im.shape[1]))

			#To capture the downsampled image
			blue_downsample_output=np.zeros(shape=(im.shape[0],im.shape[1]))
			green_downsample_output=np.zeros(shape=(im.shape[0],im.shape[1]))
			red_downsample_output=np.zeros(shape=(im.shape[0],im.shape[1]))
			
			#To capture the blurred image
			blue_blur_output=np.zeros(shape=(im.shape[0],im.shape[1]))
			green_blur_output=np.zeros(shape=(im.shape[0],im.shape[1]))
			red_blur_output=np.zeros(shape=(im.shape[0],im.shape[1]))

			#Process the test images per channel wise
			count = 0
			while count < channels:
				image=im[:,:,count]
				# blocks of original input high resolution test image
				patches_H=view_as_blocks(image,(H_patch,H_patch))
				
				image_blur=image#ndimage.gaussian_filter(image,sigma=3)
				
				#Rescaling the input test image to predict
				image_down=np.zeros(shape=(int(image.shape[0]/2),int(image.shape[1]/2)))
				image_down=misc.imresize(image_blur,((int(image.shape[0]/2),int(image.shape[1]/2))))
				patches_blur=view_as_blocks(image_down,(L_patch,L_patch))

				#Create the data set in required format for knn prediction
				i,j,k=0,0,0
				test_data = np.zeros(shape=(patches_blur.shape[0]*patches_blur.shape[1],H_patch))
				while i < patches_blur.shape[0]:
					j = 0
					while j < patches_blur.shape[1]:
						test_data[k] = np.resize(patches_blur[i][j],(1,H_patch)).astype(type)
						k=k+1
						j=j+1
					i=i+1
				
				#Bicubic interpolated data of the test image and patch wise extraction
				image_inter=misc.imresize(image_down,(int(image.shape[0]),int(image.shape[1])),"bicubic")
				patches_lup=view_as_blocks(image_inter,(H_patch,H_patch))

				ch=clf.predict(test_data)
			
				ch_rec=np.resize(ch,(patches_blur.shape[0],patches_blur.shape[1],H_patch,H_patch))

				# Add the KNN output for the bicubic interpolated
				rec_h=patches_lup.astype(type)+ch_rec.astype(type)
				rec_image=rec_h.transpose(0,2,1,3).reshape(-1,rec_h.shape[1]*rec_h.shape[3])
				if (np.amin(rec_image) < 0):
					#print("np.amin",np.amin(rec_image),np.amax(rec_image))
					rec_image=rec_image - np.amin(rec_image)
				if (np.amax(rec_image) > 255):
					#print("np.amax",np.amax(rec_image))
					rec_image = rec_image * 255.0/np.amax(rec_image)
				rec_image=np.uint8(rec_image)

				if count == 0:
					blue_output=rec_image
					blue_downsample_output=image_down
					blue_blur_output=image_blur

				if count == 1:
					green_output=rec_image
					green_downsample_output=image_down
					green_blur_output=image_blur

				if count == 2:
					red_output=rec_image
					red_downsample_output=image_down			
					red_blur_output=image_blur

				count = count + 1

			rgb_output = np.dstack((blue_output,green_output,red_output))

			fname = os.path.join(output_path, x)
			print("output file name \n",fname)
			misc.imsave(fname,rgb_output)

			rgb_input = np.dstack((blue_blur_output,green_blur_output,red_blur_output))
			y = "input_image"+x
			fname = os.path.join(output_path, y)
			misc.imsave(fname,rgb_input)

			PSNR=compare_psnr(rgb_input,rgb_output)

			rgb_downsample = np.dstack((blue_downsample_output,green_downsample_output,red_downsample_output))
			y = "down_sample_input"+x
			fname = os.path.join(output_path, y)
			misc.imsave(fname,rgb_downsample)
			print("PSNR\n",PSNR)
