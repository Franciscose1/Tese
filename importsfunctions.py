import os
import sys
import cv2
import glob
import time
import pandas
import random
import warnings
import datetime
import numpy as np
import matplotlib.pyplot as plt

from skimage.util import crop
from skimage import img_as_ubyte
from skimage.measure import label
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix,balanced_accuracy_score


import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical, Sequence 
from tensorflow.keras.losses import Loss, Reduction
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.metrics import TruePositives, FalsePositives,TrueNegatives,FalseNegatives
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.layers import Input,Dropout,Lambda,MaxPooling2D,Concatenate, Activation, Reshape
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,UpSampling2D,Cropping2D,Dense, Flatten, BatchNormalization

from metrics import *
from postprocessing import *

IMG_WIDTH = 572
IMG_HEIGHT = 572
IMG_WIDTH2 = 388
IMG_HEIGHT2 = 388
IMG_CHANNELS = 3


def import_datasetBSrot(DATAPATH,DATAPATH_segm, DATAPATH_csv,size):
    X = np.zeros((size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y = np.zeros((size, IMG_HEIGHT2, IMG_WIDTH2,1), dtype=np.float32)
    Yc = np.zeros((size,3))
    nomefich = []
    count=0
    data = pandas.read_csv(DATAPATH_csv)
    for root, dirs, files in os.walk(DATAPATH):
        for filename in files:
            img = imread(DATAPATH+filename)
            X[count] = img
            n = filename.rstrip(".jpg")
            m = filename.rstrip("r.jpg")
            nomefich.append(n + '.png')
            mask_ = imread(DATAPATH_segm+n + '.png')
            mask_ = np.expand_dims(mask_, axis=-1)
            mask_=mask_/255.0
            Y[count] = mask_
            
            Yc[count,0]=int(data[data['image_id']==m]['melanoma'])
            Yc[count,1]=int(data[data['image_id']==m]['seborrheic_keratosis'])
            Yc[count,2]=not(Yc[count,1] or Yc[count,0])
            count=count+1
    x=X
    y=Y
    yc=Yc
    print("yes")
    return x,y,yc,nomefich

def treat_datasetBSrot(DATAPATH,DATAPATH_segm, DATAPATH_csv,size):
    DATA_PATH_OUT0='ISIC2017/ISIC-2017_Training_Data/ISIC-2017_Training_Data_Resize_572_Augment_junto/0/'
    DATA_PATH_OUT0_S='ISIC2017/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth_Resize_572_Augment_junto/0/'
    DATA_PATH_OUT1='ISIC2017/ISIC-2017_Training_Data/ISIC-2017_Training_Data_Resize_572_Augment_junto/1/'
    DATA_PATH_OUT1_S='ISIC2017/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth_Resize_572_Augment_junto/1/'
    DATA_PATH_OUT2='ISIC2017/ISIC-2017_Training_Data/ISIC-2017_Training_Data_Resize_572_Augment_junto/2/'
    DATA_PATH_OUT2_S='ISIC2017/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth_Resize_572_Augment_junto/2/'
    
    Yc = np.zeros((size,3))
    nomefich = []
    count=0
    data = pandas.read_csv(DATAPATH_csv)
    for root, dirs, files in os.walk(DATAPATH):
        for filename in files:
            img = imread(DATAPATH+filename)
            n = filename.rstrip(".jpg")
            m = filename.rstrip("r.jpg")
            mask_ = imread(DATAPATH_segm+n + '.png')

            
            Yc[count,0]=int(data[data['image_id']==m]['melanoma'])
            Yc[count,1]=int(data[data['image_id']==m]['seborrheic_keratosis'])
            Yc[count,2]=not(Yc[count,1] or Yc[count,0])
            if Yc[count,0] == 1:
                imsave(DATA_PATH_OUT0+n+'.jpg',img_as_ubyte(img))
                imsave(DATA_PATH_OUT0_S+n+'.png',img_as_ubyte(mask_))
            if Yc[count,1] == 1:
                imsave(DATA_PATH_OUT1+n+'.jpg',img_as_ubyte(img))
                imsave(DATA_PATH_OUT1_S+n+'.png',img_as_ubyte(mask_))
            if Yc[count,2] == 1:
                imsave(DATA_PATH_OUT2+n+'.jpg',img_as_ubyte(img))
                imsave(DATA_PATH_OUT2_S+n+'.png',img_as_ubyte(mask_))
                
            count=count+1
    x=X
    y=Y
    yc=Yc
    print("yes")
    return x,y,yc,nomefich





def plot_performance(results):
    plt.plot(results.history['loss'], label='train loss')
    plt.plot(results.history['val_loss'], label='validation loss')
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.title('Learning Curve')
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();
    plt.show()
    
    plt.plot(results.history['jaccard'], label='trainjacc')
    plt.plot(results.history['val_jaccard'], label='testjacc')
    plt.plot(np.argmax(results.history["val_jaccard"]), np.max(results.history["val_jaccard"]), marker="x", color="r", label="best model")    
    plt.legend()
    plt.title('Intersection-Over-Union along Epochs ')
    plt.show()
    
    plt.plot(results.history['dice'], label='traindice')
    plt.plot(results.history['val_dice'], label='testdice')
    plt.legend()
    plt.title('Dice along Epochs ')
    plt.show()
    
    #plt.plot(results.history['recall'], label='train')
    #plt.plot(results.history['val_recall'], label='test')
    #plt.legend()
    #plt.title('Recall along Epochs ')
    #plt.show()
    return

def plot_performanceJunto(results):
    plt.plot(results.history['loss'], label='train loss')
    plt.plot(results.history['val_loss'], label='validation loss')
    plt.title('Average Loss')
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();
    plt.show()
    
    plt.plot(results.history['classification_output_loss'], label='train loss')
    plt.plot(results.history['val_classification_output_loss'], label='validation loss')
    plt.title('Classification Loss')
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();
    plt.show()
    
    plt.plot(results.history['segmentation_output_loss'], label='train loss')
    plt.plot(results.history['val_segmentation_output_loss'], label='validation loss')
    plt.title('Segmentation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();
    plt.show()
    
    plt.plot(results.history['segmentation_output_jaccard'], label='trainjacc')
    plt.plot(results.history['val_segmentation_output_jaccard'], label='testjacc')
    plt.plot(np.argmax(results.history["val_segmentation_output_jaccard"]), np.max(results.history["val_segmentation_output_jaccard"]), marker="x", color="r", label="best model")    
    plt.legend()
    plt.title('JAC along Epochs ')
    plt.show()
    
 
    
    plt.plot(results.history['classification_output_accuracy'], label='trainCacc')
    plt.plot(results.history['val_classification_output_accuracy'], label='testCacc')
    plt.plot(np.argmax(results.history["val_classification_output_accuracy"]), np.max(results.history["val_classification_output_accuracy"]), marker="x", color="r", label="best model")    
    plt.legend()
    plt.title('Balanced Accuracy along Epochs ')
    plt.show()  
    
    
    zipped_lists = zip(results.history['classification_output_accuracy'] , results.history['segmentation_output_jaccard'])
    sum1 = [x + y for (x, y) in zipped_lists]
    plt.plot(sum1 , label='train')
    zipped_lists2 = zip(results.history['val_classification_output_accuracy'] , results.history['val_segmentation_output_jaccard'])
    sum2 = [x + y for (x, y) in zipped_lists2]
    plt.plot(sum2, label='testCacc')  
    plt.legend()
    plt.title(' Total Accuracy along Epochs ')
    plt.show()  

    return

def plot_performanceClassification(results):
    plt.plot(results.history['loss'], label='train loss')
    plt.plot(results.history['val_loss'], label='validation loss')
    plt.title('Learning Curve')
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend();
    plt.show()
    
    plt.plot(results.history['categorical_accuracy'], label='train')
    plt.plot(results.history['val_categorical_accuracy'], label='validation')
    plt.title('Learning Curve')
    plt.plot( np.argmax(results.history["val_categorical_accuracy"]), np.max(results.history["val_categorical_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("categorical_accuracy")
    plt.legend();
    plt.show()
    
    plt.plot(results.history['accuracy'], label='train')
    plt.plot(results.history['val_accuracy'], label='validation')
    plt.title('Learning Curve')
    plt.plot( np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Weighted_accuracy")
    plt.legend();
    plt.show()
    return



def redeUNETclass(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, act='relu',complexity=64,fully=256,pool=3,dropout_c=True,initializer='he_normal'):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #DownPath
    #s = Lambda(lambda x: x / 255) (inputs)#255
    c1 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (inputs)
    c1 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c1)
    cr1 = Cropping2D(cropping=(88)) (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p1)
    c2 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c2)
    cr2 = Cropping2D(cropping=(40)) (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p2)
    c3 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c3)
    cr3 = Cropping2D(cropping=(16)) (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p3)
    c4 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c4)
    cr4 = Cropping2D(cropping=(4)) (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    #Bottleneck
    c5 = Conv2D(complexity*16, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p4)
    #c5 = Dropout(0.3) (c5)
    c5 = Conv2D(complexity*16, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c5)

    p6 = MaxPooling2D(pool_size=(pool,pool)) (c5)
    p7 = Flatten() (p6)
    if dropout_c:
        p7 = Dropout(0.5) (p7)
    p8 = Dense(fully, activation='relu') (p7)
    if dropout_c:
        p8 = Dropout(0.5) (p8)
    outputs = Dense(3,activation='softmax') (p8)

    model = Model(inputs=[inputs], outputs=[outputs])#weighted_bce
    return model

def redeUNETconjunta(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, act='relu',complexity=64,fully=256,pool=3,dropout_c=True,initializer='he_normal'):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #DownPath
    #s = Lambda(lambda x: x / 255) (inputs)#255
    c1 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (inputs)
    c1 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (c1)
    cr1 = Cropping2D(cropping=(88)) (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (p1)
    c2 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (c2)
    cr2 = Cropping2D(cropping=(40)) (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (p2)
    c3 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (c3)
    cr3 = Cropping2D(cropping=(16)) (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (p3)
    c4 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (c4)
    cr4 = Cropping2D(cropping=(4)) (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    #Bottleneck
    c5 = Conv2D(complexity*16, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(complexity*16, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (c5)

    p6 = MaxPooling2D(pool_size=(pool,pool)) (c5)
    p7 = Flatten() (p6)
    if dropout_c:
        p7 = Dropout(0.4) (p7)
    p8 = Dense(fully, activation='relu') (p7)
    if dropout_c:
        p8 = Dropout(0.4) (p8)
    classification = Dense(3,activation='softmax', name='classification_output') (p8)

    
    #Uppath
    u6 = Conv2DTranspose(complexity*8, (2, 2), strides=(2, 2), padding='valid') (c5)
    u6 = Concatenate() ([u6, cr4])
    #u6 = Concatenate([u6, cr4])
    c6 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (u6)
    c6 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (c6)

    u7 = Conv2DTranspose(complexity*4, (2, 2), strides=(2, 2), padding='valid') (c6)
    u7 = Concatenate()([u7, cr3])
    c7 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (u7)
    c7 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (c7)

    u8 = Conv2DTranspose(complexity*2, (2, 2), strides=(2, 2), padding='valid') (c7)
    u8 = Concatenate() ([u8, cr2])
    c8 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (u8)
    c8 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (c8)

    u9 = Conv2DTranspose(complexity, (2, 2), strides=(2, 2), padding='valid') (c8)
    u9 = Concatenate() ([u9, cr1])
    c9 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (u9)
    c9 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer=initializer, padding='valid') (c9)

    #segmentation = Conv2D(1, (1, 1), activation='sigmoid', name = 'segmentation_output') (c9)
    c10 = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    segmentation = Reshape((150544,1), name = 'segmentation_output') (c10)

    model = Model(inputs = [inputs], outputs = [classification, segmentation])#weighted_bce
    return model



def redeUNETclassfreeze(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, act='relu',complexity=64,fully=256,pool=3):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #DownPath
    #s = Lambda(lambda x: x / 255) (inputs)#255
    c1 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (inputs)
    c1 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c1)
    cr1 = Cropping2D(cropping=(88)) (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p1)
    c2 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c2)
    cr2 = Cropping2D(cropping=(40)) (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p2)
    c3 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c3)
    cr3 = Cropping2D(cropping=(16)) (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p3)
    c4 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c4)
    cr4 = Cropping2D(cropping=(4)) (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    #Bottleneck
    c5 = Conv2D(complexity*16, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p4)
    #c5 = Dropout(0.3) (c5)
    outputs = Conv2D(complexity*16, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c5)

    model = Model(inputs=[inputs], outputs=[outputs])#weighted_bce
    return model
    
def redeUNETclassfreeze2(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, act='relu',complexity=64,fully=256,pool=3):   
    
    inputs = Input((28, 28, 512))
    p6 = MaxPooling2D(pool_size=(pool,pool)) (inputs)
    p7 = Flatten() (p6)
    p7 = Dropout(0.5) (p7)
    p8 = Dense(fully, activation='relu') (p7)
    p8 = Dropout(0.5) (p8)
    outputs = Dense(3,activation='softmax') (p8)
    model = Model(inputs=[inputs], outputs=[outputs])#weighted_bce
    return model
    
def redeUNET(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, act='relu',complexity=64,dropout_c=True):
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #DownPath
    #s = Lambda(lambda x: x / 255) (inputs)#255
    c1 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (inputs)
    c1 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c1)
    cr1 = Cropping2D(cropping=(88)) (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p1)
    c2 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c2)
    cr2 = Cropping2D(cropping=(40)) (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p2)
    c3 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c3)
    cr3 = Cropping2D(cropping=(16)) (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p3)
    c4 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c4)
    cr4 = Cropping2D(cropping=(4)) (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    #Bottleneck
    c5 = Conv2D(complexity*16, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (p4)
    if dropout_c:
        c5 = Dropout(0.3) (c5)
    c5 = Conv2D(complexity*16, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c5)

    #Uppath
    u6 = Conv2DTranspose(complexity*8, (2, 2), strides=(2, 2), padding='valid') (c5)
    u6 = Concatenate() ([u6, cr4])
    #u6 = Concatenate([u6, cr4])
    c6 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (u6)
    c6 = Conv2D(complexity*8, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c6)

    u7 = Conv2DTranspose(complexity*4, (2, 2), strides=(2, 2), padding='valid') (c6)
    u7 = Concatenate()([u7, cr3])
    c7 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (u7)
    c7 = Conv2D(complexity*4, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c7)

    u8 = Conv2DTranspose(complexity*2, (2, 2), strides=(2, 2), padding='valid') (c7)
    u8 = Concatenate() ([u8, cr2])
    c8 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (u8)
    c8 = Conv2D(complexity*2, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c8)

    u9 = Conv2DTranspose(complexity, (2, 2), strides=(2, 2), padding='valid') (c8)
    u9 = Concatenate() ([u9, cr1])
    c9 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (u9)
    c9 = Conv2D(complexity, (3, 3), activation=act, kernel_initializer='he_normal', padding='valid') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])#weighted_bce
    return model



def create_dictionary(DATAPATH_csv,DATAPATH_segm_csv):
    import csv
    ft = open(DATAPATH_csv,'r')
    fv = open(DATAPATH_segm_csv,'r')
    readerv = csv.reader(fv)
    readert = csv.reader(ft)
    partition = {}
    labels = {}
    a=0
    for row in readert:
        if "train" not in partition:
            partition["train"]=[row[0]]
        else:
            partition["train"].append(row[0])  
        if int(float(row[1])) == 1:
            labels[row[0]] = 0
        elif int(float(row[2])) == 1:
            labels[row[0]] = 1
        else:
            labels[row[0]] = 2
    for row in readerv:
        if "validation" not in partition:
            partition["validation"]=[row[0]]
        else:
            partition["validation"].append(row[0])  
        if int(float(row[1])) == 1:
            labels[row[0]] = 0
        elif int(float(row[2])) == 1:
            labels[row[0]] = 1
        else:
            labels[row[0]] = 2 
                
    return partition, labels

#from keras.utils import to_categorical                                           
class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels, batch_size=8, n_classes=3, shuffle=True, validation = False):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        
        self.on_epoch_end()

        
        
        if validation:
            self.classe_weight = {0 : 1.66666667, 1 : 1.19047619, 2 : 0.64102564 }
            data_gen_args = dict(
                     )
            data_gen_args2 = dict(
                     )
        else:
            self.classe_weight = {0 : 1.78253119, 1 : 2.62467192, 2 : 0.48590865 }
            data_gen_args = dict(
                     horizontal_flip=True,
                     vertical_flip=True,
                    )
            data_gen_args2 = dict(
                     horizontal_flip=True,#rescale=1./255,
                     vertical_flip=True,
                    )
            
        self.image_datagen = ImageDataGenerator(**data_gen_args)
        self.mask_datagen = ImageDataGenerator(**data_gen_args2)
        seed = 1

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))                                           
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, sample_weight = self.__data_generation(list_IDs_temp)

        return X, y, sample_weight

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        DATAPATH='ISIC2017/Tudojunto_r/imagens/'
        DATAPATH2='ISIC2017/Tudojunto_r/segm/'
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X  = np.empty((self.batch_size,572, 572, 3 ))
        #ys = np.empty((self.batch_size,388, 388, 1 ))
        ys = np.empty((self.batch_size,150544, 1 ))
        yc = np.empty((self.batch_size,3))

        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Generate Transform
            transform_i = self.image_datagen.get_random_transform((572, 572, 3),seed = 1)
            #print(transform_i)
            transform_s = self.mask_datagen.get_random_transform((388, 388, 1),seed = 1)
            #print(transform_s)
            # Store sample
            img = imread(DATAPATH + ID + '.jpg')
            img=img/255
            #print(np.amax(img))
            mask = np.expand_dims(imread(DATAPATH2 + ID + '.png'), axis=-1)
            mask=mask/255
            #img2 = self.image_datagen.apply_transform(img , transform_parameters=transform_i)
            #print(np.amax(img2))
            X[i,] = self.image_datagen.apply_transform(img , transform_parameters=transform_i).astype(np.float32)
            #ys[i,] = self.mask_datagen.apply_transform(mask , transform_parameters=transform_s)
            mask2 = self.mask_datagen.apply_transform(mask , transform_parameters=transform_s).astype(np.float32)
            ys[i,] =np.reshape(mask2, (150544,1))
            #print((np.reshape(ys[i,], (150544,1))).shape)
            #imshow(img)
            #plt.show()
            #imshow(img2)
            #plt.show()
            #print(img.shape)
            #X[i,] = img
            #ys[i,] = np.expand_dims(imread(DATAPATH2 + ID + '.png'), axis=-1)
            # Store class
            yc[i] = to_categorical(self.labels[ID],3)
            
        
        # Generate Weights
        rounded_train=np.argmax(yc, axis=1)
        #print("rounded_train:",rounded_train.shape,rounded_train)
        #print("yc",yc)
        weight_class = np.array([self.classe_weight[cls] for cls in rounded_train])
        #print("weight_class",weight_class)
        #mask_ones = np.ones(388,388,1)
        #weight_segm = np.ones(((self.batch_size),388,388))
        #tf.expand_dims(weight_mask,axis=0)
        #weight_segm = np.ones(((self.batch_size),388,388,1))
        #weight_segm = tf.expand_dims(weight_segm,axis=0)
        #weight_segm = np.ones(((self.batch_size),1))
       # weight_segm = np.array([self.classe_weight[cls] for cls in rounded_train])
        weight_segm =np.ones(((self.batch_size),150544))
        
        return X, [yc,ys], [weight_class,weight_segm]                                       
                                           
class DataGenerator_segmentation(Sequence):

    
    
    def __init__(self, list_IDs, labels, batch_size=8, n_classes=3, shuffle=True, validation = False):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        
        self.on_epoch_end()     
        if validation:
            self.classe_weight = {0 : 1.66666667, 1 : 1.19047619, 2 : 0.64102564 }
            data_gen_args = dict(
                     )
            data_gen_args2 = dict(
                     )
        else:
            self.classe_weight = {0 : 1.78253119, 1 : 2.62467192, 2 : 0.48590865 }
            data_gen_args = dict(
                     horizontal_flip=True,
                     vertical_flip=True,
                    )
            data_gen_args2 = dict(
                     horizontal_flip=True,#rescale=1./255,
                     vertical_flip=True,
                    )
            
        self.image_datagen = ImageDataGenerator(**data_gen_args)
        self.mask_datagen = ImageDataGenerator(**data_gen_args2)
        seed = 1
    

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))                                           
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        DATAPATH='ISIC2017/Tudojunto_r/imagens/'
        DATAPATH2='ISIC2017/Tudojunto_r/segm/'
        #DATAPATH='ISIC2017/Tudojunto/imagens/'
        #DATAPATH2='ISIC2017/Tudojunto/segm/'
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X  = np.empty((self.batch_size,572, 572, 3 ))
        ys = np.empty((self.batch_size,388, 388, 1 ))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Generate Transform
            transform_i = self.image_datagen.get_random_transform((572, 572, 3),seed = 1)
            transform_s = self.mask_datagen.get_random_transform((388, 388, 1),seed = 1)
            # Store sample
            img = imread(DATAPATH + ID + '.jpg')
            img=img/255
            mask = np.expand_dims(imread(DATAPATH2 + ID + '.png'), axis=-1)
            mask=mask/255

            X[i,] = self.image_datagen.apply_transform(img , transform_parameters=transform_i).astype(np.float32)
            ys[i,] = self.mask_datagen.apply_transform(mask , transform_parameters=transform_s).astype(np.float32)
     
        return X,ys                                      
                                           