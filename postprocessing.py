import os
from matplotlib import pyplot as mlt
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from skimage import img_as_ubyte
import numpy as np
from skimage.util import crop
import matplotlib.pyplot as plt
import cv2
import time
from skimage.measure import label 



def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def closeit(imagePredicted,K):
    kernel = np.ones((K,K),np.uint8)
    bin_img = imagePredicted*255
    bin_img=bin_img.astype(np.uint8)
    imageSe = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)#Morphologic close
    return imageSe
    

#https://github.com/SherylHYX/Scan-flood-Fill
background_threshold = 50
# pixel with difference less than this value with the background colour is seen as background colour

def del_file(path):
    # delete files
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def getIOU(imageSe, imageTrue):
    intersection = np.logical_and(imageSe, imageTrue)
    union = np.logical_or(imageSe, imageTrue)
    iou_score_Pos = np.sum(intersection) / np.sum(union)
    return iou_score_Pos



def getIOU_DATAPATH(DATAPATH_output,DATAPATH_original):
    for root, dirs, files in os.walk(DATAPATH_original):
        iou_score=0
        a=0
        for filename in files:
            imageTrue =imread(DATAPATH_original+filename)
            #filenameS = filename.rstrip(".jpg")
            #imageTrue = imread(DATAPATH_original+filenameS+'.png')#TRUE
            #imagePredicted = imread(DATAPATH_output+filenameS+'.png')#PREDICTED
            imagePredicted = imread(DATAPATH_output+filename)#PREDICTED
            iou_score=iou_score + getIOU(imagePredicted, imageTrue)
            a=a+1 
    return iou_score/a 

def readAndPad(imagePath, backGroundColour):
    img = cv2.imread(imagePath, 0)
    padImg = np.pad(img,1,pad_with, padder=backGroundColour)
    height, width = padImg.shape[:2]
    return padImg, height, width

def floodFill(img, height, width, x, y, boundaryColour, backGroundColour, fillColour):
    img[x, y] = fillColour
    if (x>0 and img[x-1, y] == backGroundColour):
        floodFill(img, height, width, x-1, y, boundaryColour, backGroundColour, fillColour)
    if (y>0 and img[x, y-1] == backGroundColour):
        floodFill(img, height, width, x, y-1, boundaryColour, backGroundColour, fillColour)
    if (x<(height-1) and img[x+1, y] == backGroundColour):
        floodFill(img, height, width, x+1, y, boundaryColour, backGroundColour, fillColour)
    if (y<(width-1) and img[x, y+1] == backGroundColour):
        floodFill(img, height, width, x, y+1, boundaryColour, backGroundColour, fillColour)

def cvFloodFill(img, height, width, fillColour):
    mask = np.zeros([height+2, width +2], np.uint8)
    cv2.floodFill(img, mask, (0, 0), newVal = fillColour, loDiff = 50, upDiff = 50, flags = 4)

def cropAndReverse(img, height, width, maskColour, backGroundColour, fillColour):
    croppedImg = np.delete((np.delete(img, [0, width-1], axis=1)), [0, height-1], axis=0)
    for x in range(height-2):
        for y in range(width-2):
            # fillColour is the colour that is temporarily used to fill exterior
            # maskColour is for the final result (interior)
            if croppedImg[x,y] == fillColour:
                croppedImg[x,y] = backGroundColour
            elif abs(croppedImg[x,y] - backGroundColour) < background_threshold:
                croppedImg[x,y] = maskColour
    return croppedImg

def fillBoundaryMain(imagePath ,savePath):

    padImg, height, width = readAndPad(imagePath, backGroundColour = 0)
    # Home made flood fill
    #floodFill(padImg, height, width, 0, 0, 255, 0, 128)

    cvFloodFill(padImg, height, width, 128)
    #cv2.imwrite(cvPath,padImg, [int(cv2.IMWRITE_JPEG_QUALITY),100])
    filledImg = cropAndReverse(padImg, height, width, 255, 0, 128)
    cv2.imwrite(savePath, filledImg, [int(cv2.IMWRITE_JPEG_QUALITY),100])
    #padImg.fill(0)
    #filledImg.fill(0)
    
    
def PostProcess(DATAPATH,DATAPATH_Segm,DATAPATH_unet,DATAPATH_OUT,DATAPATH_OUT_2):
    for root, dirs, files in os.walk(DATAPATH):
        iou_score=0
        iou_scorepost=0
        a=0
        for filename in files:
            image =imread(DATAPATH+filename)
            filenameS = filename.rstrip(".jpg")
            imageTrue = imread(DATAPATH_Segm+filenameS+'.png')#TRUE
            imagePredicted = imread(DATAPATH_unet+filenameS+'.png')#PREDICTED
            #imshow(imageTrue)
            #plt.show()
            image=closeit(imagePredicted,9)
            #imshow(image)
            #plt.show()
            imageCC=getLargestCC(image)
            #imshow(imageCC)
            #plt.show()
            iou_scorepost=iou_scorepost + getIOU(imageCC, imageTrue)
            iou_score=iou_score + getIOU(imagePredicted, imageTrue)
            imsave(DATAPATH_OUT+filenameS+'.png',img_as_ubyte(imageCC))
            a=a+1
            #break
        #break
        iou_scorepost = iou_scorepost/a     
        iou_score = iou_score/a   
        print('Before Processing: iou=',iou_score)
        print('After Processing: iou=',iou_scorepost)

    input_path = DATAPATH_OUT
    output_path_fill = DATAPATH_OUT_2
    # delete previous files
    #os.mkdir(output_path_fill)
    del_file(output_path_fill)
    for root, dirs, files in os.walk(input_path):
        for f in files:
            fillBoundaryMain(os.path.join(input_path, f), os.path.join(output_path_fill, f))



    for root, dirs, files in os.walk(DATAPATH):
        iou_score=0
        iou_scorepost=0
        a=0
        for filename in files:
            image =imread(DATAPATH+filename)
            filenameS = filename.rstrip(".jpg")
            imageTrue = imread(DATAPATH_Segm+filenameS+'.png')#TRUE
            imagePredicted = imread(DATAPATH_OUT+filenameS+'.png')#PREDICTED
            imagePredicted2 = imread(DATAPATH_OUT_2+filenameS+'.png')#PREDICTED
            iou_score=iou_score + getIOU(imagePredicted, imageTrue)
            iou_scorepost=iou_scorepost + getIOU(imagePredicted2, imageTrue)
            a=a+1
        iou_scorepost = iou_scorepost/a   
        print('After Flood Filling: iou=',iou_scorepost)
        iou_score = iou_score/a
        #print('Best Case Scenario: iou=',iou_score)

def save_predictions(DATAPATH_OUT,preds,nomefich):
    a=0
    preds_vale=np.where(preds > 0.5,255,0)
    for image in preds_vale:
        imsave(DATAPATH_OUT+nomefich[a],img_as_ubyte(image))
        a=a+1
    return
def return_to_life(DATAPATH_output,DATAPATH_segm,DATAPATH_output2):
    for root, dirs, files in os.walk(DATAPATH_output):
        for filename in files:
            #Abre ficheiro doo utput e faz decrop 
            img = imread(DATAPATH_output+filename)
            temp_imgS = np.pad(img[:,:],((92,92),(92,92)),'constant', constant_values = (0))
            #Abre ficheiro original para confirmar tamanhos
            filenameS = filename.rstrip(".png")
            imgO = imread(DATAPATH_segm+filenameS+'_segmentation.png')
            height, width = imgO.shape
            if width > height:
                width2=width+(184*2)
                height2=width2
            else:
                height2=height+(184*2)
                width2=height2     
            imageS = resize(temp_imgS,(height2, width2),  anti_aliasing=True)
            if width > height:
                width2=width+(184*2)
                add_x = int((width2-height)/2)#cima
                add_x2=width2-height-add_x # baixo
                add_y=92*2 #esquerda
                add_y2=92*2 #direita

                resized_imgS = crop(imageS, ((add_x,add_x2), (add_y,add_y2)), copy=False)
            else:
                eight2=height+(184*2)
                add_y=int((height2-width)/2)#esquerda
                add_y2=height2-width-add_y#direita
                add_x=92*2 #cima
                add_x2=92*2#baixo
                resized_imgS = crop(imageS, ((add_x,add_x2), (add_y,add_y2)), copy=False)
            imsave(DATAPATH_output2+filenameS+'_segmentation.png',img_as_ubyte(resized_imgS))

        
def automatePostProcess(preds_val_t,DATAPATH_OUT,nomefichv):
    save_predictions(DATAPATH_OUT,preds_val_t,nomefichv)
    
    DATAPATH_output=DATAPATH_OUT
    DATAPATH_original='ISIC2017/ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth_Resize_5721/'
    print('IOU=', getIOU_DATAPATH(DATAPATH_output,DATAPATH_original))
    
    DATAPATH = 'ISIC2017/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data_Resize_5721/'
    DATAPATH_Segm = 'ISIC2017/ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth_Resize_5721/'
    DATAPATH_unet = 'ISIC2017/ISIC-2017_Validation_Part1_GroundTruth/Result/teste/1/'
    DATAPATH_OUT = 'ISIC2017/ISIC-2017_Validation_Part1_GroundTruth/Result/teste/1post/'
    DATAPATH_OUT_2 = 'ISIC2017/ISIC-2017_Validation_Part1_GroundTruth/Result/teste/1postEFCI/'
    PostProcess(DATAPATH,DATAPATH_Segm,DATAPATH_unet,DATAPATH_OUT,DATAPATH_OUT_2)
    
    DATAPATH_output = DATAPATH_OUT_2
    DATAPATH_output2 = 'ISIC2017/ISIC-2017_Validation_Part1_GroundTruth/Result/teste/1final/'
    DATAPATH_Segm = 'ISIC2017/ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth/'
    return_to_life(DATAPATH_output,DATAPATH_Segm,DATAPATH_output2)
    
    DATAPATH_output = DATAPATH_output2
    DATAPATH_original = 'ISIC2017/ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth/'
    print('Final IOU=', getIOU_DATAPATH(DATAPATH_output,DATAPATH_original))
    
    
    
    
def automatePostProcessTest(preds_val_t,DATAPATH_OUT,nomefichv):
    save_predictions(DATAPATH_OUT,preds_val_t,nomefichv)
    
    DATAPATH_output=DATAPATH_OUT
    DATAPATH_original='ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC-2017_Test_v2_Part1_GroundTruth_Resize_5721/'
    print('IOU=', getIOU_DATAPATH(DATAPATH_output,DATAPATH_original))
    
    DATAPATH = 'ISIC2017/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data_Resize_5721/'
    DATAPATH_Segm = 'ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC-2017_Test_v2_Part1_GroundTruth_Resize_5721/' 
    DATAPATH_unet = 'ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth/Result/teste/1/'
    DATAPATH_OUT = 'ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth/Result/teste/1post/'
    DATAPATH_OUT_2 = 'ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth/Result/teste/1postEFCI/'
    PostProcess(DATAPATH,DATAPATH_Segm,DATAPATH_unet,DATAPATH_OUT,DATAPATH_OUT_2)
    
    DATAPATH_output = DATAPATH_OUT_2
    DATAPATH_output2 = 'ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth/Result/teste/1final/'
    DATAPATH_Segm = 'ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC-2017_Test_v2_Part1_GroundTruth/'
    return_to_life(DATAPATH_output,DATAPATH_Segm,DATAPATH_output2)
    
    DATAPATH_output = DATAPATH_output2
    DATAPATH_original = 'ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC-2017_Test_v2_Part1_GroundTruth/'
    print('Final IOU=', getIOU_DATAPATH(DATAPATH_output,DATAPATH_original))