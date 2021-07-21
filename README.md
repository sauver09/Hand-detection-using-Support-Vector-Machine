# 

<h1> Object detection using Support Vector Machine</h1>

In this project, we train an Support Vector Machine(SVM) and use it for detecting human hands in iamges. We use deep features extracted from detectron2 (https://github.com/facebookresearch/detectron2). To detect human hands in images, we need a classifier that can distinguish between hand image patches from non-hand patches. To train such a classifier, we can use SVMs. The training data is typically a set of images with bounding boxes of the hands. Positive training examples are image patches extracted at the annotated locations. A negative training example can be any image patch that does not significantly overlap with the annotated hands. Thus there potentially many more negative training examples than positive training examples. Due to memory limitation, it will not be possible to use all negative training examples at the same time. So, we implement hard-negative mining to find hardest negative examples and iteratively train an SVM.

## Steps to Run:
* Make sure you have following libraries installed
  * mean_average_precision
  * PyYAML
  * matplotlib
  * scipy
  * sklearn
  * cv2
  
## Prerequisites
``` 
!pip install mean average precision
!git clone https://github.com/facebookresearch/detectron2.git --branch v0.1.1 detectron2_v0.1.1
%cd detectron2_v0.1.1/
git checkout db1614e
!pip install -e .


Inside the detectron2 v0.1.1/ directory, unzip the given HW4 q3.zip:
!unzip HW4q3.zip %cd HW4 q3

Under HW4 q3/, there are 2 python files: hw4 utils.py that contains helper functions and detect.py that includes the feature extraction with detectron2.

Then, download the ContactHands dataset inside HW4 q3/ directory from http://vision.cs.stonybrook. edu/ ̃supreeth/ContactHands_data_website/ or by running:
!wget https://public.vinai.io/ContactHands.zip ! unzip ContactHands
The ContactHands/README.md provides useful information regarding the structure of this dataset.


```

## Two important Functions:
<h3> Hard_negative_mining(feat_extractor,D_train,lb_train,D_test,lb_test) </h3>

```

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
 ```
 
<h3> gatherHardestNegative(clf,feat_extractor,number_of_image_to_consider) </h3>

```
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
```

Note: 
The baseline code is taken from ContactHands(http://vision.cs.stonybrook.edu/~supreeth/ContactHands_data_website/) and the above two functions are my contribution.
