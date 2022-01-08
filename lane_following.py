#!/usr/bin/env python

from __builtin__ import True # for python2
import numpy as np
import rospy
import math
import torch
import rospkg
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torchvision
import cv2
import os
from torchvision import transforms, utils, datasets
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

bridge = CvBridge()
weight_path = '/home/sis/locobot_competition/catkin_ws/src/baseline_navi/weight/yb_lane.pth'


"""""

This Part You Have To Build Your Own Model

For example:
class CNN_Model(nn.Module)

"""""


class Lane_follow(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial()
        self.omega = 0
        self.count = 0

        self.data_transform = transforms.Compose([transforms.ToTensor()]) 
        # motor omega output (the number of rotation, from L_1 to S_R)
        self.Omega = np.array([0.1,0.17,0.24,0.305,0.37,0.44,0.505,0.73,-0.1,-0.17,-0.24,-0.305,-0.37,-0.44,-0.505,-0.73,0.0,0.0])
        rospy.loginfo("[%s] Initializing " % (self.node_name))
        self.pub_car_cmd = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=1)
        self.pub_cam_tilt = rospy.Publisher("/tilt/command", Float64, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_cb, queue_size=1)
    
    # load weight
    def initial(self):
        self.model = CNN_Model()
        self.model.load_state_dict(torch.load(weight_path))
        self.model = self.model.to(self.device)

       
    # load image to define omega for motor controlling
    def img_cb(self, data):
        #self.dim = (101, 101)  # (width, height)
        self.count += 1
        self.pub_cam_tilt.publish(0.9)
        if self.count == 6:
            self.count = 0
            try:
                # convert image_msg to cv format
                img = bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")

                img = self.data_transform(img)
                images = torch.unsqueeze(img,0)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                images = images.to(self.device)
                self.model = self.model.to(self.device)
                output = self.model(images)
                top1 = output.argmax()
                self.omega = self.Omega[top1]
                
                # motor control
                car_cmd_msg = Twist()
                car_cmd_msg.linear.x = 0.123
                car_cmd_msg.angular.z = self.omega*0.64
                
                self.pub_car_cmd.publish(car_cmd_msg)
                
                rospy.loginfo('\n'+str(self.omega)+'\n'+str(top1))

            except CvBridgeError as e:
                print(e)



if __name__ == "__main__":
    rospy.init_node("lane_follow", anonymous=False)
    lane_follow = Lane_follow()
    rospy.spin()