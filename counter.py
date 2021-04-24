import cv2
import mediapipe as mp
import time
import os
import numpy as np
import math

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

kypt_index = { 'NOSE' : 0, 'LEFT_EYE_INNER' : 1, 'LEFT_EYE' : 2, 'LEFT_EYE_OUTER' : 3, 'RIGHT_EYE_INNER' : 4, 'RIGHT_EYE' : 5, 'RIGHT_EYE_OUTER' : 6, 
'LEFT_EAR' : 7, 'RIGHT_EAR' : 8, 'MOUTH_LEFT' : 9, 'MOUTH_RIGHT' : 10, 'LEFT_SHOULDER' : 11, 'RIGHT_SHOULDER' : 12, 'LEFT_ELBOW' : 13, 'RIGHT_ELBOW' : 14, 
'LEFT_WRIST' : 15, 'RIGHT_WRIST' : 16, 'LEFT_PINKY' : 17, 'RIGHT_PINKY' : 18, 'LEFT_INDEX' : 19, 'RIGHT_INDEX' : 20, 'LEFT_THUMB' : 21, 'RIGHT_THUMB' : 22, 
'LEFT_HIP' : 23, 'RIGHT_HIP' : 24, 'LEFT_KNEE' : 25, 'RIGHT_KNEE' : 26, 'LEFT_ANKLE' : 27, 'RIGHT_ANKLE' : 28, 
'LEFT_HEEL' : 29, 'RIGHT_HEEL' : 30, 'LEFT_FOOT_INDEX' : 31, 'RIGHT_FOOT_INDEX' : 32}

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, rhs):
        return Point(self.x+rhs.x,self.y+rhs.y)
    
    def __sub__(self, rhs):
        return Point(self.x-rhs.x,self.y-rhs.y)
    
    def __mul__(self, rhs):
        return self.x*rhs.x+self.y*rhs.y

    def size(self):
        return (self.x**2+self.y**2)**0.5
    

def degree(pt1, pt2, pt3):
    d1 = pt1-pt2
    d2 = pt3-pt2
    inner_product = d1*d2
    return math.acos(inner_product/d1.size()/d2.size())


class jumpingjack_counter:
    joint = ['LEFT_SHOULDER','LEFT_ELBOW','LEFT_WRIST','LEFT_HIP','LEFT_KNEE','LEFT_ANKLE',
            'RIGHT_SHOULDER','RIGHT_ELBOW','RIGHT_WRIST','RIGHT_HIP','RIGHT_KNEE','RIGHT_ANKLE']

    angle_pair_list = [('LEFT_SHOULDER','LEFT_ELBOW','LEFT_WRIST'),('RIGHT_SHOULDER','RIGHT_ELBOW','RIGHT_WRIST'),
    ('LEFT_HIP','LEFT_SHOULDER','LEFT_ELBOW'),('RIGHT_HIP','RIGHT_SHOULDER','RIGHT_ELBOW'),
    ('LEFT_SHOULDER','LEFT_HIP','LEFT_KNEE'),('RIGHT_SHOULDER','RIGHT_HIP','RIGHT_KNEE'),
    ('LEFT_HIP','LEFT_KNEE','LEFT_ANKLE'),('RIGHT_HIP','RIGHT_KNEE','RIGHT_ANKLE')]

    pca_axis = np.load('jj_pca.npy').T

    center = np.load('jj_center.npy')
    
    def __init__(self):
        self.detection = []

    def cal_angle(self, detection):
        j_det = np.zeros(len(jumpingjack_counter.angle_pair_list))
        for i, angle_label in enumerate(jumpingjack_counter.angle_pair_list):
            pts = [Point(detection[kypt_index[angle_label[j]]].x, detection[kypt_index[angle_label[j]]].y) for j in range(3)]
            j_det[i] = degree(pts[0],pts[1],pts[2])
        
        return j_det
    
    def PCA(self, detection):
        return np.matmul(detection, jumpingjack_counter.pca_axis)
    
    def get_cluster(self, detection):
        j_det = self.cal_angle(detection)
        pt = self.PCA(j_det)
        l = 10000000000
        c_ind = 0
        for i, ct in enumerate(jumpingjack_counter.center):
            d = ct - pt
            if d[0]**2+d[1]**2 < l:
                c_ind = i
                l = d[0]**2+d[1]**2
        return c_ind

    def add(self, detection):
        self.detection.append(self.get_cluster(detection))

    def save(self, name):
        np.save(name,self.detection)

vid_name = 'jumpingjack_1.mp4'
os.makedirs(vid_name[:-4],exist_ok=True)
cap = cv2.VideoCapture('jumpingjack_1.mp4')
pTime = time.time()
counter = jumpingjack_counter()
num = 0
prev = -1
while True:
    num+=1
    success, img = cap.read()
    if num > 12640:
        break
    if num < 790:
        continue
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        counter.add(results.pose_landmarks.landmark)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    if prev != counter.detection[-1]:
        print(counter.detection[-1])
        prev = counter.detection[-1]
    cv2.putText(img, str(counter.detection[-1]),(100,100), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 3)
    cv2.imshow('Image',img)
    
    cv2.waitKey(100)

counter.save('jumpingjack_1_final')