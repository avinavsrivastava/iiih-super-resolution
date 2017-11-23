from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from scipy import misc
from scipy import ndimage
import numpy
import cv2
import os
import timeit

# Initalization parameters
DATA_PATH = "train/"
TEST_PATH = "test/"
VAL_PATH = "val/"

ACTIVATION_FUN = 'linear'
EPOCH_VALUE = 10
POOL_VALID = 1
SCALE = 2

# Constants
Random = 30
Patch_size = 32
label_size = 20

# Training Data and Validation data collection
def prepare_data(_path):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()
    
    data = numpy.zeros((nums * Random, 1, Patch_size, Patch_size), dtype=numpy.double)
    label = numpy.zeros((nums * Random, 1, label_size, label_size), dtype=numpy.double)
    
    for i in range(nums):
        name = _path + names[i]
        if name.endswith(".jpg"):
          hr_img = misc.imread(name,mode='RGB')
          shape = hr_img.shape
                    
          hr_img = hr_img[:, :, 0]
          hr_img = ndimage.gaussian_filter(hr_img,sigma=3)
                            
          # two resize operation to produce training data and labels
          lr_img = misc.imresize(hr_img, (shape[0] / SCALE, shape[1] / SCALE))
          lr_img = misc.imresize(lr_img, (shape[0], shape[1]), "bicubic")
                                    
          # produce random coordinate to crop training img
          Points_x = numpy.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random)
          Points_y = numpy.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random)
                                            
          for j in range(Random):
            lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
            hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
                                                        
            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.
            
                                                                
            data[i * Random + j, 0, :, :] = lr_patch
            label[i * Random + j, 0, :, :] = hr_patch[6: -6, 6: -6]
    return data, label

# Super Resolution Convolution model with covn-pool layers and compile with MSE loss function
def model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=128, kernel_size = (9,9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(Patch_size, Patch_size, 1)))
    if POOL_VALID == 1:
      SRCNN.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    SRCNN.add(Conv2D(filters=64, kernel_size = (3,3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    if POOL_VALID == 1:
      SRCNN.add(UpSampling2D(size=(2,2)))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5,5), kernel_initializer='glorot_uniform',
                     activation= ACTIVATION_FUN , padding='valid', use_bias=True))
    sgd = SGD(lr=0.0003)
    SRCNN.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN

# Model for Prediction/regression, Image as an input
def predict_model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=128, kernel_size = (9,9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (3,3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5,5), kernel_initializer='glorot_uniform',
                     activation= ACTIVATION_FUN, padding='valid', use_bias=True))
    sgd = SGD(lr=0.0003)
    SRCNN.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN

# Train model with lowering mean squared error loss
def train(data, label,val_data, val_label):
    srcnn_model = model()

    # Loss minimization
    checkpoint = ModelCheckpoint("SRCNN_weights.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, epochs=EPOCH_VALUE, verbose=0)

# Prediction of Image
def predict_image(_path):
    srcnn_model = predict_model()
    srcnn_model.load_weights("SRCNN_weights.h5")

    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()
    pre_input = 0
    pre_output = 0
    for i in range(nums):
      name = _path + names[i]
      if name.endswith(".jpg"):
          hr_img = misc.imread(name,mode='RGB')
          IMG_NAME = name
          INPUT_NAME = os.path.splitext(name)[0] + "_input.jpg"
          BICUBIC_NAME = os.path.splitext(name)[0] + "_bicubic.jpg"
          OUTPUT_NAME = os.path.splitext(name)[0] + "_predicted.jpg"
          
          img = misc.imread(IMG_NAME, mode='RGB')
          shape = img.shape
          Y_img = misc.imresize(img[:, :, 0], (shape[0] / SCALE, shape[1] / SCALE))
    
          Y_img = misc.imresize(Y_img, (shape[0], shape[1]), "nearest")
          img[:, :, 0] = Y_img
          misc.imsave(INPUT_NAME, img)
          Y_img = misc.imresize(Y_img, (shape[0], shape[1]), "bicubic")

          img[:, :, 0] = Y_img
          misc.imsave(BICUBIC_NAME, img)

          Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
          Y[0, :, :, 0] = Y_img.astype(float) / 255.
          pre = srcnn_model.predict(Y, batch_size=1) * 255.
    
          pre = pre.astype(numpy.uint8)
          img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
          misc.imsave(OUTPUT_NAME, img)

          # psnr calculation:
          im1 = misc.imread(INPUT_NAME)
          im2 = misc.imread(OUTPUT_NAME)
          imHR = img
              
          input_PSNR = cv2.PSNR(imHR, im1)
          output_PSNR = cv2.PSNR(imHR, im2)
          
          if pre_output < output_PSNR:
            pre_input = input_PSNR
            pre_output = output_PSNR
            demo = IMG_NAME
  
    print "PSNR HR - INPUT"
    print pre_input
    print "PSNR HR - OUTPUT"
    print pre_output
    print demo


if __name__ == "__main__":
    
    start = timeit.default_timer()

    print "POOL_VALID", POOL_VALID
    print "ACTIVATION_FUN", ACTIVATION_FUN
    print "EPOCH_VALUE", EPOCH_VALUE
    print "SCALE", SCALE

    data, label = prepare_data(DATA_PATH)
    val_data, val_label = prepare_data(VAL_PATH)
    data = numpy.transpose(data, (0, 2, 3, 1))
    label = numpy.transpose(label, (0, 2, 3, 1))
    val_data = numpy.transpose(val_data, (0, 2, 3, 1))
    val_label = numpy.transpose(val_label, (0, 2, 3, 1))

    train(data, label,val_data, val_label)
    predict_image(TEST_PATH)

    stop = timeit.default_timer()
    print stop - start
