#https://gist.github.com/ilmonteux/8340df952722f3a1030a7d937e701b5a
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix,balanced_accuracy_score
import numpy as np

def weighted_bce(y_true, y_pred):
    pos_weight = 1
    x_1 = y_true * pos_weight * -K.log(y_pred + 1e-6)
    x_2 = (1 - y_true) * -K.log(1 - y_pred + 1e-6)
    return (x_1 + x_2)

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def jaccard(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    jaccard = true_positives/(possible_positives + predicted_positives - true_positives + K.epsilon())
    return jaccard

def dice(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    dice = (2*(true_positives))/(possible_positives + predicted_positives + K.epsilon())
    return dice

def classification_metrics(predictions,groundtruth):
    groundtruth_arg=np.argmax(groundtruth, axis=1)
    predictions_arg=np.argmax(predictions, axis=1)
    print("Prediction confusion Matrix")
    print(confusion_matrix(groundtruth_arg,predictions_arg))
    print("Original confusion Matrix")
    print(confusion_matrix(groundtruth_arg,groundtruth_arg))
    
    confus=confusion_matrix(groundtruth_arg,predictions_arg)

    #Melanoma:
    TPm=confus[0][0]
    FNm=confus[0][1]+confus[0][2]
    FPm=confus[1][0]+confus[2][0]
    TNm=confus[1][1]+confus[1][2]+confus[2][1]+confus[2][2]
    #SK:
    TPsk=confus[1][1]
    FNsk=confus[1][0]+confus[1][2]
    FPsk=confus[0][1]+confus[2][1]
    TNsk=confus[0][0]+confus[0][2]+confus[2][0]+confus[2][2]
    #Nevus:
    TPn=confus[2][2]
    FNn=confus[2][0]+confus[2][1]
    FPn=confus[0][2]+confus[1][2]
    TNn=confus[0][0]+confus[0][1]+confus[1][0]+confus[1][1]

    Sensivitym=TPm/(TPm+FNm)
    Sensivityn=TPn/(TPn+FNn)
    Sensivitysk=TPsk/(TPsk+FNsk)
    print('Sensivity[M N SK]:',"{:.3f}".format(Sensivitym),'.',"{:.3f}".format(Sensivityn),'.',"{:.3f}".format(Sensivitysk))

    Specificitym=TNm/(TNm+FPm)
    Specificityn=TNn/(TNn+FPn)
    Specificitysk=TNsk/(TNsk+FPsk)
    print('Specificit[M N SK]:',"{:.3f}".format(Specificitym),'.',"{:.3f}".format(Specificityn),'.',"{:.3f}".format(Specificitysk))

    Balanced_Accuracy=balanced_accuracy_score(groundtruth_arg,predictions_arg)
    print('Balanced_Accuracy by scikit:',"{:.3f}".format(Balanced_Accuracy))

    Categorical_Accuracy=((TPm+TPsk+TPn)/(TPm+TPsk+TPn+confus[0][1]+confus[0][2]+confus[1][0]+confus[1][2]+confus[2][0]+confus[2][1]))##Categorical Accuracy
    print('Categorical Accuracy:',"{:.3f}".format(Categorical_Accuracy))
    
    return

