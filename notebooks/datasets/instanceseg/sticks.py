import cv2
import numpy as np
import math

class RRect_center:
    def __init__(self, p0, s, ang):
        (self.W, self.H) = s # rectangle width and height
        self.d = math.sqrt(self.W**2 + self.H**2)/2.0 # distance from center to vertices    
        self.c = (int(p0[0]+self.W/2.0),int(p0[1]+self.H/2.0)) # center point coordinates
        self.ang = ang # rotation angle
        self.alpha = math.radians(self.ang) # rotation angle in radians
        self.beta = math.atan2(self.H, self.W) # angle between d and horizontal axis
        # Center Rotated vertices in image frame
        self.P0 = (int(self.c[0] - self.d * math.cos(self.beta - self.alpha)), int(self.c[1] - self.d * math.sin(self.beta-self.alpha))) 
        self.P1 = (int(self.c[0] - self.d * math.cos(self.beta + self.alpha)), int(self.c[1] + self.d * math.sin(self.beta+self.alpha))) 
        self.P2 = (int(self.c[0] + self.d * math.cos(self.beta - self.alpha)), int(self.c[1] + self.d * math.sin(self.beta-self.alpha))) 
        self.P3 = (int(self.c[0] + self.d * math.cos(self.beta + self.alpha)), int(self.c[1] - self.d * math.sin(self.beta+self.alpha))) 

        self.verts = [self.P0,self.P1,self.P2,self.P3]

    def draw(self, image):
        rect = cv2.minAreaRect(np.asarray(self.verts))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,0,-1)
        cv2.drawContours(image,[box],0,255,2)
        
    def fill_value(self, image, value):
        rect = cv2.minAreaRect(np.asarray(self.verts))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,value,-1)

def sticks_instance_segmentation_sample(width, height, max_sticks=5):

    img = np.zeros((height, width)).astype(np.uint8)
    mask = np.zeros((height, width, 1)).astype(np.uint8)
    n_sticks = np.random.randint(2, max_sticks)
    for i in range(1, n_sticks+1):
        
        top_left = (np.random.randint(20, width-20), np.random.randint(20, height-20)) 
        angles = np.random.randint(0, 360)
        
        rect = RRect_center(top_left, (10, 50), angles)
        while not all([(0 < x[0] < width) and (0 < x[1] < height) for x in rect.verts]):
            top_left = (np.random.randint(20, width-20), np.random.randint(20, height-20)) 
            rect = RRect_center(top_left, (10, 50), angles)
        
        rect.draw(img)
        rect.fill_value(mask, i)
        
        
    return np.expand_dims(img, -1), mask