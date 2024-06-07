import cv2
import numpy as np
import time
import queue
import threading
from pymycobot.mycobot import MyCobot

class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
    
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except Queue.Empty:
                    pass
            self.q.put((ret, frame))
            
    def read(self):
        return self.q.get()
                

relocate = 0
miss_threshold = 60

speed = 0.5
error = 20 # 15
radius_error = 6
basic_angles = (0,63,-30,0,40,0) #(0,70,-35,0,40,0)
put_down_angles = (0,50,-30,0,40,0)
angles = list(basic_angles)
gripper_open = 100
gripper_close = 60

mc = MyCobot('/dev/ttyAMA0',1000000)
def arm_initial():
    mc.power_on()
    mc.set_fresh_mode(1)
    mc.set_gripper_value(gripper_open, 15)
    time.sleep(2)
    mc.send_angles(angles,40)
    time.sleep(3)
    print("initial finish")

def arm_exit():
    mc.stop()
    #mc.release_all_servos()
    

   
def arm_adjust_total(ret, x, y, z, curr_x, curr_y, curr_z):
    global angles
    if ret is False:
        #time.sleep(0.1)
        print("false")
        if mc.is_moving():
            print("moving")
            #mc.jog_stop()
            #time.sleep(0.1)
            return 0
        return 0

    error_x = curr_x - x
    error_y = curr_y - y
    error_z = curr_z - z

    if abs(error_x) > error or abs(error_y) > error or abs(error_z) > radius_error:
        direction_x = -0.5 if (error_x > 0) else 0.5
        direction_y = -0.4 if (error_y > 0) else 0.4 #best
        direction_z1 = -0.5 if (error_z > 0) else 0.5
        direction_z2 = -0.5 if (error_z < 0) else 0.5
        #print(error_x, error_y,error_z)
        
        if abs(error_x) > error and abs(error_y) > error:
            angles[4] += direction_x
            angles[0] += direction_y
        elif abs (error_x) > error:
            angles[4] += direction_x
        elif abs (error_y) > error:
            angles[0] += direction_y
        if abs(error_z) > radius_error:
            angles[1] += direction_z1
            if angles[1] > 90:
                angles[1] = 90
                return 1
            angles[2] += direction_z2
            if angles[2] < -90:
                angles[2] = -90

        #mc.send_angles(angles,1)
        #time.sleep(0.05)
        return 1
    else:
        print("minimal")
        #time.sleep(0.1)
        if mc.is_moving():
            print("moving")
            #mc.jog_stop()
            #time.sleep(0.1)
        return 2
        



    
def circle_detect(image=None):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    low_hsv = np.array([12, 100, 100])
    high_hsv = np.array([35, 255, 255])
    
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

    cv2.imshow("mask", mask)    
    
    dst = cv2.blur(mask, (1, 16))
    
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 40, param1=150, param2=25, minRadius=30, maxRadius=150)
    
    pos = [0, 0, 0]
    
    if circles is not None:
        for i in circles[0, :]: 
            pos[0] = int(i[0])
            pos[1] = int(i[1])
            pos[2] = int(i[2])
            ret = True
    else:
        ret = False
    return ret, pos
    
cap = VideoCapture(0)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
#ret, frame = cap.read()

#ret_code = cap.grab()
#ret, image = cap.retrieve()

#ret, pos = circle_detect(frame)
#video_ret, frame = cap.read()
#frame_count = 0
#fps = 1

arm_initial()



#average = pos[2]
#count = 0
while True:
    
    #cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    ret, frame = cap.read()
    #cap.release()
    if True:
        ball_ret, pos = circle_detect(frame)
        

        
        ret = arm_adjust_total(ball_ret, 160, 260, 128, pos[0], pos[1], pos[2])
        print(angles)
        if ret == 2:
            count += 2
            if count >= 2:
                #cap.release()
                print("gripping!!!!!!!!!!!!!!!!")
                time.sleep(0.2)
                #mc.release_all_servos()
                
                mc.set_gripper_value(gripper_close, 20)
                time.sleep(1)
                angles = list(put_down_angles)  # put down ball
                mc.sync_send_angles(angles,50, timeout=2) #sync addition
                time.sleep(2)
                mc.set_gripper_value(gripper_open, 20)
                time.sleep(1)
                angles = list(basic_angles)
                print(basic_angles,"!!!!!!!!!!!!")
                mc.sync_send_angles(angles,50, timeout=2) #sync addition
                
                count = 0
                #cap = cv2.VideoCapture(0)
                #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        elif ret == 1:
            count = 0
            mc.sync_send_angles(angles, 1, timeout=1) #sync addition
            
            #time.sleep(0.2)
        else:
            count = 0
            relocate += 1
            if relocate >= miss_threshold:
                relocate = 0
                angles = list(basic_angles)
                mc.sync_send_angles(angles, 40, timeout=1)
            

             
        
        cv2.circle(frame, (pos[0], pos[1]), 2, (0, 0, 0), 2)
        

        cv2.imshow("frame", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        #frame_count = 0 
    #video_ret, frame = cap.read()
        
cap.release()
arm_exit()
