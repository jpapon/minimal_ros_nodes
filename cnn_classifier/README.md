# Setting up CNN cnn_classifier
## Dependencies
### Random python dependencies
1. Install pip and python-dev 
 `sudo apt-get install python-pip python-dev`
2. Install other python dependencies using pip
 `sudo pip install numpy scipy pillow matplotlib commentjson`

### Install CUDA and CUDNN (Optional)
1. Follow steps one and two here:
 [CUDA and CUDNN Install](http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html)
 
### Install Tensorflow
1. First, update pip:
 `sudo pip install --upgrade pip`
2. Install the CPU version of tensorflow:
 `sudo pip install tensorflow`
3. If you have a GPU, install the gpu version (this requires a CUDA install)
 `sudo pip install tensorflow-gpu`
4. It can also be useful to make sure that all dependencies are the most recent version by doing an upgrade:
 `sudo pip install --upgrade tensorflow`
 
### Install TensorVision and Dependencies
1. Install deps
 `sudo pip install -r https://raw.githubusercontent.com/TensorVision/TensorVision/master/requirements.txt`
2. Install tensorvision
 `sudo pip install https://github.com/TensorVision/TensorVision/archive/master.zip`

### Try it out.
  A demo launch file which runs the classifier and shows the labeled image is provided:
  `roslaunch cnn_classifier cnn_classifier.launch playback_path:=PATH_TO_BAG_FILES`

#Example outputs with labeled GT and color legend
![CNN Examples](qualitative_cnn.png "CNN Examples")

# Label Colors:
"unlabeled_color" : [0,0,0,0],
"grass_color" : [1,0,255,0],
"foliage_color" : [2,0,128,0],
"wood_color" : [3,128,  32,  64],
"dirt_color" : [4,128,  32,  32],
"pavement_color" : [5, 32,  32,  128],
"rock_color" : [6,75,  75,  75],
"water_color" : [7, 0,  0,  255],
"sky_color" : [8, 157,  250,  255],
"building_color" : [9,255,  0,  0],


# Training
For training, you'll need to checkout my Github Repo, which is a fork of MarvinTeichmann's KittiSeg
[Fork of KittiSeg](https://github.com/jpapon/KittiSeg)

## Prepare Data
  To prepare the data for training, there's a Python script  KittiSeg/data_prep/prepare_data.py.
  To use this, you specify a DATA_DIR, and then the two folders within it (IMAGE_FOLDER, GT_FOLDER)
  which contain the images and their associated GT images. GT images here are RGB labeled images, 
  with one color for each label. These colors & labels are specified in the hyperparameter file (more on that later).
  This script will go through, divide the dataset randomly into a training and validation set 
  (by default 5% of the images will go to validation - this can be set with VAL_PROP), and then create
  two files - TRAIN_FILE_OUT and VAL_FILE_OUT, which contain the names of the input images and their
  ground truth labels for training and validation.
  
  These two files should then be placed in DATA_DIR, where they will be read from during training.
  The paths to them can be specified in the hypes file, but this default location of DATA_DIR should work with the current hypes files.
  
  Training is all based around hypes files - you can find some of them in KittiSeg/hypes. If you wish to train a network on new data, it's easiest to copy one of the existing hypes files, and modify it suit your needs.
  
  
  Training can be run by two simple commands. The first sets the output working directory:
  `export TV_DIR_RUNS=OUPUT_DIR`
  where OUTPUT_DIR specifies the folder which will contain the output network weights and validation image visualizations.
  Then the actual training is run with:
  `tv-train --hypes hypes/SensorFusionSeg.json`
  
  
  
  
  


