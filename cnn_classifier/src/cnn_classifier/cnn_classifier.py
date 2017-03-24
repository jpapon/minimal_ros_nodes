#!/usr/bin/env python
# Python libs
import sys, time, os

# numpy 
import numpy as np
from scipy.misc import imsave

# OpenCV
import cv2

import tensorflow as tf
import tensorvision.utils as tv_utils
import tensorvision.core as core

# Ros libraries
import rospy
import rospkg

# Ros Messages
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class cnn_classifier:

    def __init__(self, save_output = False):
        '''Initialize ros publisher, ros subscriber'''
        self.save_output_ = save_output
        # topic where we publish RGB version of classified result
        self.pub_rgb_labels = rospy.Publisher("rgb_labels",Image,queue_size=1)
        self.pub_labels = rospy.Publisher("labels",Image,queue_size=1)
        
        self.bridge = CvBridge()
        
        # subscribed Topic
        self.subscriber = rospy.Subscriber("image", Image, self.image_callback,
                                           queue_size=1)
        
        rospy.loginfo ("Initializing Network...")
        rospack = rospkg.RosPack()
        network_path = rospack.get_path('cnn_weights')
        self.network_path_ = os.path.join(network_path , 'networks','segnet')
        self.hypes_ = tv_utils.load_hypes_from_logdir(self.network_path_)
        self.num_classes_ = self.hypes_['arch']['num_classes']
        rospy.loginfo("Hypes loaded successfully.")
        # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
        self.modules_ = tv_utils.load_modules_from_logdir(self.network_path_)
        rospy.loginfo("Modules loaded, building tf graph.")

        # Create tf graph and build module.
        with tf.Graph().as_default():
            # Create placeholder for input
            self.image_pl_ = tf.placeholder(tf.float32)
            image = tf.expand_dims(self.image_pl_, 0)
            
            self.hypes_['dirs']['data_dir'] = self.network_path_
            # build Tensorflow graph using the model from logdir
            self.prediction_ = core.build_inference_graph(self.hypes_, self.modules_,
                                                    image=image)

            rospy.loginfo("Graph built successfully.")

            # Create a session for running Ops on the Graph.
            self.sess_ = tf.Session()
            self.saver_ = tf.train.Saver()

            # Load weights from logdir
            core.load_weights(self.network_path_, self.sess_, self.saver_)

            rospy.loginfo("Weights loaded successfully.")

        #Build map for colorizing
        self.label_colors_ = {}
        self.label_colors_alpha_ = {}
        for key in self.hypes_['data'].keys():
            if ('_color' in key):
                color = np.array(self.hypes_['data'][key])
                self.label_colors_[color[0]] = (color[1],color[2],color[3],255)
                self.label_colors_alpha_[color[0]] = (color[1],color[2],color[3],128)
    
        rospy.logwarn ("Done loading neural network!")
        
        self.image_count_ = 0
        self.output_dir_ = "CNN_test_output"
        if not os.path.exists(self.output_dir_):
            os.makedirs(self.output_dir_)
            
    #---------------------------------------------------------------------------
    # Image subscriber callback.
    #---------------------------------------------------------------------------
    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            rospy.logerr( "Error =" + e)
    
        label_im, label_overlay_im = self.classify_image(cv_image)
        if (self.save_output_):
            imsave (os.path.join(self.output_dir_,
                                 "input_{:05d}.png".format(self.image_count_)),
                    cv_image)
            imsave (os.path.join(self.output_dir_,
                                 "overlay_{:05d}.png".format(self.image_count_)),
                    label_overlay_im)
            rospy.loginfo ("Saved frame {:05d}".format (self.image_count_))
        self.image_count_ += 1
        
        try:
            self.pub_rgb_labels.publish(self.bridge.cv2_to_imgmsg(label_overlay_im, "rgb8"))
            self.pub_labels.publish(self.bridge.cv2_to_imgmsg(label_im, "mono8"))
        except CvBridgeError as e:
            rospy.logerr( "Error =" + e)
        
        
    def classify_image (self, input_img):
        start = time.time()
        #rospy.loginfo ("Classifying image size={}".format(input_img.shape))

        # Run KittiSeg model on image
        feed = {self.image_pl_: input_img}
        softmax = self.prediction_['softmax']
        output = self.sess_.run([softmax], feed_dict=feed)

        # Reshape output from flat vector to 2D Image
        shape = input_img.shape
        output = output[0].reshape(shape[0], shape[1], self.num_classes_)
        label_im = np.argmax(output, axis = 2).astype(np.uint8)
        label_overlay_im = tv_utils.overlay_segmentation(input_img, label_im, self.label_colors_alpha_)

        rospy.loginfo ("Time to run neural net on image = {:.3f}s".format(time.time()-start))

        return label_im, label_overlay_im

def main(args):
    
    '''Initializes and cleanup ros node'''
    rospy.init_node('cnn_classifier', anonymous=True)
    save_output = rospy.get_param('save_output', False)
    classifier = cnn_classifier(save_output = save_output)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down CNN Classifier module")

if __name__ == '__main__':
    main(sys.argv)