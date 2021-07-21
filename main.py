from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

clf = LinearSVC(loss='hinge',random_state=42, tol=1e-5,max_iter=1000)
clf.fit(D_train,lb_train)
y_pred=clf.predict(D_test)
acc=accuracy_score(lb_test, y_pred)
print("Accuracy:",acc)

generate_result_file(feat_extractor,clf,"validation")
ap=compute_mAP("result.npy", "validation")

from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(clf, D_test, lb_test)

 
from sklearn.svm import LinearSVC


#A helper function which returns weight,bias,LinearSVC estiamtor and support indexes
#####################################
# Input :
# D_train : n*d array which consists of features of n images
# lb_train: 1*n array which consists of the label associated 
# max_iter: number of iteration 

# Output:
# w: weight  
# b: bias
# svm_clf: LinearSVC model
# support_vector_indices: support indexes 
#####################################
def trainSVM(D_train,lb_train,max_iter=1000):
  svm_clf = LinearSVC(loss='hinge',random_state=42, tol=1e-5,max_iter=max_iter)
  svm_clf.fit(D_train, lb_train)
  b = svm_clf.intercept_
  w = svm_clf.coef_
  
  decision_function = svm_clf.decision_function(D_train)
  support_vector_indices = np.where(
      np.abs(decision_function) <= 1 + 1e-15)[0]
  
  return w,b,svm_clf,support_vector_indices


  #A helper function which is used to calculate primal objective value for SVM
#####################################
# Input :
# X : n*d array which consists of features of n images
# y: 1*n array which consists of the label associated 
# w: weight  
# b: bias
# C: hyper parameter to control the degree of error you allow w.r.t decision boundary

# Output:

# primal_obj: objective value calculated for primal problem for SVM
#####################################

def cal_objective_primal(X, y, w, b, C):
    primal_obj = 1- y*(np.dot(X, np.transpose(w)+b))
    primal_obj[primal_obj < 0] = 0
    L = np.sum(primal_obj) * C
    primal_obj = (0.5) * np.dot(w, np.transpose(w)) + L
    return primal_obj

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

#A helper function which is used to calculate accuracy, objective and average precision of the model
#####################################
# Input :
# feat_extractor: feat_extractor which is initialised as part of the problem which helps in extracting features from image
# clf: LinearSVC model
# D_train : n*d array which consists of features of n training examples
# lb_train: 1*n array which consists of the label associated 
# D_test : m*d array which consists of features of m test examples
# lb_test: 1*m array which consists of the label associated with m test examples
# w: weight  
# b: bias
# supportIndices: support indexes of SVM
# C: hyper parameter to control the degree of error you allow w.r.t decision boundary

# Output:

# obj : objective value for primal problem of SVM
# acc : Accuracy obtained when tried predicting with input model clf
# ap : average precision
#####################################

def calculate_accuracy_objective_and_ap(feat_extractor,clf,D_train,lb_train,D_test,lb_test,w,b,supportIndices,C=1):
  obj=cal_objective_primal(D_train, lb_train, w, b, C)
  print("#########################")
  y_pred=clf.predict(D_test)
  acc=accuracy_score(lb_test, y_pred)
  print("Accuracy:",acc)
  print("Objective value: ",obj)
  ap = average_precision_score(lb_test, y_pred)
  print("Ap val: ",ap)
  print("#########################")
 
  ###########################
 
  return obj,acc,ap;

#A helper function which is used to get the annotation names and its dataset which can be used later during hard-Mining
#####################################
# Input :
#dataset: either "train" or "validation"

# Output:

# ann_names= List of annotation names
# dataset= dataset corresponding to the annotations
#####################################

def load_ann_names_from_ImageSets_AnyDataset_TextFile(dataset="train"):
  ann_names=[]
  dataset_file = "ContactHands/ImageSets/Main/{}.txt".format(dataset)
  with open(dataset_file, "r") as f:
      dataset = f.read().splitlines()
  
  for ann_name in tqdm.tqdm(dataset):
    ann_names.append(ann_name)
  return ann_names,dataset

reg_size = (32, 32)
image_size = (256, 256)
import random

####Note: I have updated the detect function just a little bit in order to get the features from there itself

# def detect(img, feat_extractor, svm_model, returnFeature=False,batch_size=-1):
#     """
#     Perform sliding window detection with SVM.
#     Return a list of rectangular regions with scores.
#     Return rects: (n, 5) where rects[i, :] corresponds to
#                   left, top, right, bottom, detection score
#     """
#     proposal_boxes = get_proposals(img.shape[0], img.shape[1])
#     n = proposal_boxes.shape[0]
#     batch_size = (n if batch_size == -1 else batch_size)
#     scores = None

#     inference_second_stream(feat_extractor, img)

#     for i in range(0, n, batch_size):
#         # start_time = time.time()
#         D = feat_extractor.extract_features(proposal_boxes[i:i+batch_size, :])
#         D = D.detach().to('cpu').numpy()
#         # print("Feature extraction for {} proposals takes {} seconds".format(proposal_boxes.shape[0], time.time() - start_time))

#         # start_time = time.time()
#         if scores is None:
#             scores = svm_model.decision_function(D)
#         else:
#             scores = np.concatenate((scores, svm_model.decision_function(D)),
#                                     axis=0)
#         # print("SVM scoring takes {} seconds".format(time.time() - start_time))

#     if returnFeature==True:
#         rects = np.concatenate((proposal_boxes, np.expand_dims(scores, axis=1),D), axis=1)
#     else:
#         rects = np.concatenate((proposal_boxes, np.expand_dims(scores, axis=1)), axis=1)

#     # Non maximum suppression
#     # start_time = time.time()
#     rects = nms(rects, overlap_thresh=0.5)
#     # print("NMS takes {} seconds".format(time.time() - start_time))

#     return rects
#################################################


#A helper function which is used to gather HardestNegative examples
#####################################
# Input :
# clf: LinearSVC model
# feat_extractor: feat_extractor which is initialised as part of the problem which helps in extracting features from image
# number_of_image_to_consider: numer of images to consider for the purpose of hard mining from training set

# Output:

# B: Hardest negative examples
#####################################

def gatherHardestNegative(clf,feat_extractor,number_of_image_to_consider=100):
 
  B=[] #Hardest negative examples

  ### getting random annotation names and its corresponding dataset
  images_pool = "ContactHands/JPEGImages/"
  Harderst_rects_list=[]
  images_pool = "ContactHands/JPEGImages/"
  annotations_pool = "ContactHands/Annotations/"
 
  ann_names,dataset=load_ann_names_from_ImageSets_AnyDataset_TextFile("train")
  random.shuffle(ann_names) #randomizing
  ann_names=ann_names[:number_of_image_to_consider]
 

  count=1;
  for ann_name in ann_names:
    # Get annotation and corresponding image
    count=count+1
    xml_path = os.path.join(annotations_pool, ann_name + ".xml")
    image_file, boxes = read_content(xml_path)
    image_path = os.path.join(images_pool, image_file)
    
    if image_file.split(".jpg")[0] in dataset:
        img = cv2.imread(image_path)
        img = cv2.resize(img, image_size)
        
        rand_rects=detect(img,feat_extractor,clf,returnFeature=True) #modeified detect function inorder to get the feature
        
        #a negative example should not have significant overlap with any annotated hand
        for box in boxes:
          iou = get_iou(box,rand_rects)
          rand_rects = rand_rects[iou < 0.3]
          if len(rand_rects) == 0:
              break
       
       #considering best 20 rectangles 

        maxlen=min(rand_rects.shape[0],20) 
        
        for Harderst_rect in rand_rects[:maxlen]:
          B.append(Harderst_rect[5:]) #appending the featues associated with each rectangle
      
  return B


import numpy as np
from detect import prepare_second_stream, inference_second_stream
from hw4_utils import get_iou,read_content,get_pos_and_random_neg,generate_result_file,compute_mAP,detect
from sklearn.metrics import accuracy_score
from sklearn import svm
import tqdm
import cv2
import os
 

#A  function which is used for Hard_negative_mining
#####################################
# Input :
# feat_extractor: feat_extractor which is initialised as part of the problem which helps in extracting features from image
# number_of_image_to_consider: numer of images to consider for the purpose of hard mining from training set
# D_train : n*d array which consists of features of n training examples
# lb_train: 1*n array which consists of the label associated 
# D_test : m*d array which consists of features of m test examples
# lb_test: 1*m array which consists of the label associated with m test examples

# Output:

# B: Hardest negative examples
# clf: LinearSVC model
# stats_obj_list : list consisting of objective values at each iteration of hard mining 
# stats_acc_list : list consisting of accuracy values at each iteration of hard mining 
# stats_ap_list  : list consisting of average precision values at each iteration of hard mining 
#####################################

def Hard_negative_mining(feat_extractor,D_train,lb_train,D_test,lb_test):

  #storing the index of positive and negative examples and indexes are from actual D_train/lb_train
  posD=[i for i,j in enumerate(lb_train) if lb_train[i]==1] #indexes of all annotated hands
  negD=[i for i,j in enumerate(lb_train) if lb_train[i]==-1] # indexes of all random image patches or not the annotated hands
 
  stats_obj_list=[];
  stats_acc_list=[];
  stats_ap_list=[];
  
  #calling trainSVM function inorder to get w,b,clf,support_vectors_idx
  w,b,clf,support_vectors_idx=trainSVM(D_train,lb_train)

  #calling calculate_accuracy_objective_and_ap function inorder to get obj,acc,ap
  obj,acc,ap=calculate_accuracy_objective_and_ap(feat_extractor,clf,D_train,lb_train,D_test,lb_test,w,b,support_vectors_idx,C=1)
  
  #updating all the three lists
  stats_obj_list.append(obj)
  stats_acc_list.append(acc)
  stats_ap_list.append(ap)
 
  #Running for 10 iteration 

  for iter in range(10):
    supportIndices=support_vectors_idx

    A=[i for i,j in enumerate(lb_train) if (j==-1) and (i not in supportIndices)] #index of All non support vectors in NegD
    B=gatherHardestNegative(clf,feat_extractor) # features corresponding to hardest negative examples
    negDMinusA=[i for i in negD if i not in A] # consist of indexes corresponding to support vectors in negD (NegD ← (NegD \ A) )
    
    startPosforB=len(D_train)
    for b_ in B:
      D_train = np.insert(D_train, D_train.shape[0],b_, axis=0)
      lb_train=np.insert(lb_train,lb_train.shape[0],-1)
      
    negD=negDMinusA+[i for i in range(startPosforB,startPosforB+len(B))]  #indexes corresponging to negD as : NegD ← (NegD \ A) ∪ B.
  
    lb_train_pos=lb_train[posD]
    D_train_pos=D_train[posD]
    lb_train_neg=lb_train[negD]
    D_train_neg=D_train[negD]
    
    lb_train=np.concatenate((lb_train_pos,lb_train_neg),axis=0)
    D_train=np.concatenate((D_train_pos,D_train_neg),axis=0)

    #reinitializing our posD and negD after adding hardest negative examples
    posD=[i for i,j in enumerate(lb_train) if lb_train[i]==1]
    negD=[i for i,j in enumerate(lb_train) if lb_train[i]==-1]
    
    #calling trainSVM function inorder to get w,b,clf,support_vectors_idx
    w,b,clf,support_vectors_idx=trainSVM(D_train,lb_train)
    #calling calculate_accuracy_objective_and_ap function inorder to get obj,acc,ap
    obj,acc,ap=calculate_accuracy_objective_and_ap(feat_extractor,clf,D_train,lb_train,D_test,lb_test,w,b,support_vectors_idx,C=1)
  
    #updating all the three lists
    stats_obj_list.append(obj)
    stats_acc_list.append(acc)
    stats_ap_list.append(ap)
    
  return clf,stats_obj_list,stats_acc_list,stats_ap_list;

#####################Calling Hard_negative_mining function################

clf,stats_obj_list,stats_acc_list,stats_ap_list=Hard_negative_mining(feat_extractor,D_train,lb_train,D_test,lb_test) #latest

obj_print=[i[0][0] for i in stats_obj_list]

import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
p1,=plt.plot(stats_ap_list,color='blue')
plt.xlabel("Number of Iteration")
plt.ylabel("Average Precision")
plt.legend(handles=[p1], title='Average Precision Vs Number of Iteration', bbox_to_anchor=(0.25, 1), loc='upper left')
plt.grid()


plt.figure(figsize=(8,8))
p1,=plt.plot(obj_print,color='blue')
plt.xlabel("Number of Iteration")
plt.ylabel("Objective function Value")
plt.legend(handles=[p1], title='Objective function Value Vs Number of Iteration (Primal Problem of SVM)', bbox_to_anchor=(0.1, 1), loc='upper left')
plt.grid()

import pickle
# save the model to disk
filename = 'finalized_model_final_sub.sav'
pickle.dump(model, open(filename, 'wb'))