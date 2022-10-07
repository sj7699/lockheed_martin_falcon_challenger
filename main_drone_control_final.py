import sys
import traceback
from djitellopy import Tello
import av
import cv2 
import numpy as np
import time
import os,sys
from threading import Thread
from time import sleep
import math
import pyzbar.pyzbar as pyzbar
import torch
from PIL import Image

drone = Tello()
modelinfo = None

def model34(model,img):
    global modelinfo
    print("detect")
    modelinfo = model(img)
    print(modelinfo.pandas().xyxy[0])
    
def simpleKalman(x0,P0,z,R):
    A = 1
    A_t = A
    H = 1
    H_t = H
    Q = 1

    xp = A*x0
    Pp = A*P0*A_t + Q

    K = (Pp*H_t)/(H*Pp*H_t + R)

    x = xp + K*(z- H*xp)
    P = Pp - K*H*Pp

    return x,P

def PD(e,e_old,del_t,K_P,K_D):

    out = e * K_P + (e - e_old) * K_D / del_t

    return out

def rotate_mat(x,y,theta):

    theta = theta * math.pi / 180
    x_r = x*math.cos(theta) - y*math.sin(theta)
    y_r = x*math.sin(theta) + y*math.cos(theta)

    return x_r, y_r

def down(time):
    global drone
    print("Down")
    drone.send_rc_control(0,0,-10,0)
    sleep(time)
    drone.send_rc_control(0,0,0,0)
    print("Down Finished")

def up(dist):
    global drone
    print("Up")
    drone.move_up(dist)
    #sleep(3)
    #drone.send_rc_control(0,0,0,0)
    print("Up Finished")

def hover():
    global drone
    drone.send_rc_control(0,0,0,0)
    sleep(7)
    print("Hovering Finished")

def mission1():
    global drone
    drone.move_back(30)
    sleep(2)
    drone.move_forward(30) # 본래 미션은 10cm이나 최소 입력이 20입니다.
    sleep(2)

def mission2():
    global drone
    drone.move_left(30)
    sleep(2)
    drone.move_right(30)
    sleep(2)

def mission3():
    global drone
    drone.rotate_clockwise(360)
    sleep(3)

def mission4():
    global drone
    drone.move_right(20)
    sleep(1)
    drone.move_up(20)
    sleep(1)
    drone.move_left(20)
    sleep(1)
    drone.move_down(20)
    sleep(2)

def mission5():
    global drone
    drone.flip_back()
    sleep(3)

def mission6():
    global drone
    drone.move_up(30)
    sleep(3)
    drone.flip_back()
    sleep(3)
    drone.move_down(30)
    sleep(2)

def mission7():
    global drone
    drone.flip_left()
    sleep(3)

def mission8():
    global drone
    drone.move_up(30)
    sleep(1)
    drone.move_down(30)
    sleep(2)

def mission9(img):
    cv2.imwrite("C:/Users/chang/Desktop/detect_image/mission9_Detected.png",img)

def digit_mission(digit):
    global drone
    print("mission " + str(digit) +" start!")
    if digit == 1:
        mission1()
    elif digit == 2:
        mission2()
    elif digit == 3:
        mission3()
    elif digit == 4:
        mission4()
    elif digit == 5:
        mission5()
    elif digit == 6:
        mission6()
    elif digit == 7:
        mission7()
    elif digit == 8:
        mission8()

def state_35():
    drone.rotate_clockwise(90)
    sleep(2)

def main():
    
    #variables definition
    global drone,digit
    is_before_mission=False
    digit_old = 0
    iter = [0,0,0,0,0]
    iter_center = 0
    filt_iter = 0
    filt_iter_d = 0
    digit = 0
    iter_digit = 0
    iter_yolo = 0
    e_x = 0
    e_y = 0
    e_yaw = 0
    e_dist = 0
    e_x_old = 0
    e_y_old = 0
    e_yaw_old = 0
    e_dist_old = 0
    P0 = np.array([6,6,6,6,6])
    P0_d = np.array([6,6,6,6,6])
    number_var =  np.array([0,0,0,0,0])
    number_detect = False
    KAUmarker_var_raw = np.array([0,0,0,0]) # xmin, ymin, xmax, ymax // confi, class_name
    KAUmarker_detect = False
    P0_3 = np.array([6,6,6,6])
    F22marker_var_raw = np.array([0,0,0,0]) # xmin, ymin, xmax, ymax // confi, class_name
    F22marker_detect = False
    P0_4 = np.array([6,6,6,6])
    detect_color = None
    color_detect_desire = None
    mission_state = 3 # which flag(color) to detect (-1: QR, 0: GREEN, 1: RED, 2: BLUE)
    mission_process = 0   # drone control process at the mission state
    is_up = False
    pic_save = True
    
    try:
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('output_num.avi',fourcc,30,(960,720))

        # Model
        model_num = torch.hub.load('E:/KAU/공모전/2022_록히드마틴/본선/yolov5', 'custom', path='E:/KAU/공모전/2022_록히드마틴/본선/number_yolov5m_4/content/yolov5/runs/train/exp/weights/best.pt', source='local')
        model_3 = torch.hub.load('E:/KAU/공모전/2022_록히드마틴/본선/yolov5', 'custom', path='E:/KAU/공모전/2022_록히드마틴/본선/marker_yolov5m(640,batch_32,epoch_150)data3/content/yolov5/runs/train/exp/weights/best.pt', source='local')
        model_4 = torch.hub.load('E:/KAU/공모전/2022_록히드마틴/본선/yolov5', 'custom', path='E:/KAU/공모전/2022_록히드마틴/본선/f22_yolov5m4/f22best.pt', source='local')
        t_model = Thread(target=model34, args=(model_3,1))

        drone.connect()
        # print("take off!")
        # sleep(3)
        drone.takeoff()
        # sleep(3)
        # print("take off finished")
        # drone.move_back(30)
        # sleep(3)
        # drone.move_down(20)
        # sleep(3)

        # video start
        drone.streamon()
        frame_read = drone.get_frame_read()

        # thread definition
        t_hover = Thread(target=hover)
        t_mission_up= Thread(target=up,args=(25,))
        t_digit = Thread(target=digit_mission)
        t_state35 = Thread(target=state_35)

        while True:
            k = False
            marker = False

            image = frame_read.frame
            #out.write(image)
            image_out = image.copy()
            #mask = np.zeros_like(image_out)

            #blur
            image_blur = cv2.GaussianBlur(image,(0,0),3,3)

            #gray
            image_gray = cv2.cvtColor(image_blur, cv2.COLOR_RGB2GRAY)
            image_gray2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            #Canny  
            img_canny = cv2.Canny(image_blur, 70, 100)

            #Dilate    
            kernel=np.ones((3,3),int)
            img_dilate=cv2.dilate(img_canny,kernel,iterations=3)

            ########################################################################################################
            ########################################### DRONE CONTROL ##############################################
            altitude = drone.get_height()

            ######################################### mission state -1 : QR Hovering #########################################
            if mission_state == -1:
                
                # QR detect 1 right after take-off
                detectqr=pyzbar.decode(image)
                if(len(detectqr)!=0):
                    print(detectqr[0].data.decode('utf-8'))
                    print("Hovering Start!")
                    if(not t_hover.is_alive()): t_hover.start()
                    mission_state = 0
                    mission_process = 0  
                else: 
                    drone.send_rc_control(0, 0, 0, 40) # rotate until find QR
            
            ####################### mission state 0,1,2 : Flag missions ################ ############################
            if mission_state >= 0 and mission_state <= 2 and not t_digit.is_alive():

                #contour Detection
                contours,hier = cv2.findContours(img_dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                #mission state 
                if mission_state == 0:   # first detect black
                    color_detect_desire = 3
                elif mission_state == 1:   # second detect red
                    color_detect_desire = 2
                elif mission_state == 2:    # last detect blue
                    color_detect_desire = 0

                area_max = 1

                for cnt in contours:

                    # ellipse detection
                    (el_cx, el_cy),(el_h,el_w),angle = cv2.fitEllipse(cnt) #타원찾는거? -> ㅇㅇ
                    area = cv2.contourArea(cnt)
                    area_el = math.pi *el_h*el_w*0.25
                    
                    # avoid division 0
                    if area_el == 0:
                        area_el = 100000
                        
                    fit_el = float(area/area_el)

                    # avoid center out of frame
                    if el_cx >= 960 or el_cx < 0 or el_cy >= 720 or el_cy < 0:
                        continue

                    #color detection / detect_color => 0 : Blue, 1 : Green , 2 : Red, 3: Black
                    detect_point = np.array([[el_cx,el_cx+20,el_cx+20,el_cx-20,el_cx-20], [el_cy,el_cy+20,el_cy-20,el_cy+20,el_cy-20]])

                    if el_cx + 20 >= 960:
                        detect_point[0,1] = el_cx 
                        detect_point[0,2] = el_cx
                    if el_cx - 20 <= 0:
                        detect_point[0,3] = el_cx
                        detect_point[0,4] = el_cx
                    if el_cy + 20 >= 720:
                        detect_point[1,1] = el_cy
                        detect_point[1,3] = el_cy
                    if el_cy - 20 <= 0:
                        detect_point[1,2] = el_cy
                        detect_point[1,4] = el_cy
                    
                    sum_color = [0,0,0]
                    v_color = [0,0,0]    

                    # color save
                    # B,G,R
                    for i in range(0,5):
                        cp = image_blur[int(detect_point[1,i]),int(detect_point[0,i])]
                        sum_color = sum_color + cp
                    
                    v_color = sum_color/5

                    detect_color = None
                    
                    if (np.max(v_color)- np.min(v_color)  >= 10):
                        detect_color = v_color.argmax()
                    
                    # Black by RGB
                    if ((np.max(v_color)-np.min(v_color) <= 10) and mission_state == 0):
                        detect_color = 3
                    
                    if fit_el >= 0.95 and area > 3000 : 
                        print(v_color, detect_color,np.max(v_color)- np.min(v_color))

                    # ellipse save -> mission state 0,1,2
                    if fit_el >= 0.95 and area > 3000 and detect_color == color_detect_desire:
                        marker = True
                        #save maximum area ellipse
                        if (area - area_max)/area_max >= 0.1:
                            area_max = area 
                            raw_data_d = np.array([el_cx,el_cy,el_h/2,el_w/2,angle])

                        raw_data = np.array([el_cx,el_cy,el_h/2,el_w/2,angle])
                        
                        #simpleKalman filter
                        if filt_iter == 0:
                            el_var_avg = raw_data
                            filt_iter = 1

                        el_var_avg, P0 = simpleKalman(el_var_avg, P0, raw_data, 144)   
    
                        #show center of the flag
                        el_center = (int(el_var_avg[0]),int(el_var_avg[1]))
                        cv2.circle(image_out,el_center,2,(0,0,0),3)
                        cv2.ellipse(image_out,(int(el_cx),int(el_cy)),(int(el_h/2),int(el_w/2)),angle,0,360,(255,255,255),2) # show all colored ellipses

                        #color str save
                        if detect_color == 0:
                            color = 'BLUE'           
                        elif detect_color == 1:
                            color = 'GREEN'               
                        elif detect_color == 2:
                            color = 'RED'
                        elif detect_color == 3:
                            color = 'Black'  

                if marker == True:             
                    #simpleKalman filter
                    if filt_iter_d == 0:
                        el_var_avg_d = raw_data_d
                        filt_iter_d = 1

                    el_var_avg_d, P0_d = simpleKalman(el_var_avg_d, P0_d, raw_data_d,9)   

                    #show max ellipse 
                    el_center_d = (int(el_var_avg_d[0]),int(el_var_avg_d[1]))
                    el_hw_d = (int(el_var_avg_d[2]),int(el_var_avg_d[3]))
                    angle_t_d = int(el_var_avg_d[4])

                    cv2.ellipse(image_out,el_center_d,el_hw_d,angle_t_d,0,360,(255,0 ,255),3)
                    cv2.circle(image_out,el_center_d,2,(255,0,255),2)
                    cv2.putText(image_out,'Size : '+str(int(el_var_avg_d[2])),(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,238,0),2)
                    cv2.putText(image_out,color,(int(el_var_avg_d[0]),int(el_var_avg_d[1]+el_var_avg_d[2])),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)

                    if pic_save == True and mission_process == 2:
                        cv2.imwrite("C:/Users/chang/Desktop/detect_image/"+str(color)+"_Detected.png",image_out)
                        pic_save = False

                ## process 1.0
                if mission_process == 0 and is_before_mission and not t_mission_up.is_alive():
                        drone.send_rc_control(0,0,0,0)
                        t_mission_up=Thread(target=up,args=(20,))
                        t_mission_up.start() 
                        is_before_mission=False
                
                if mission_process == 0 and not t_hover.is_alive() and not t_mission_up.is_alive(): # rotate counter-clockwise until find the marker
                    iter[0] = iter[0] + 1
                    iter[1] = 0 
                    iter[3] = 0
                    iter[4] = 0
                    iter_digit = 0

                    if iter[0] == 10:
                        drone.send_rc_control(0, 0, 0, -40) 
                        iter[0] = 0
                    if marker:
                        mission_process = 1
                    marker = False

                ## process 1.1
                elif mission_process == 1: # PD control make center of the ellipse be center of the image frame
                    
                    if marker == False:
                        iter[0] = iter[0] + 1

                        if iter[0] == 8: # if no marker detected for 8 frames -> go back to mission process 0 
                            mission_process = 0

                    else:
                        iter[1] = iter[1] + 1
                        iter[0] = 0

                        if iter[1] == 10:
                            drone.send_rc_control(0, 0, 0, -15) # first inertia
                            e_x = 0
                            e_y = 0
                            e_x_old = 0
                            e_y_old = 0

                        if iter[1] == 19:
                            drone.send_rc_control(0, 0, 0, 0) 

                        if iter[1] == 25: # 5 iter -> 0.1s

                            del_t = 0.1

                            K_P = 0.05
                            K_D = 0.03

                            #error save
                            e_x = -el_var_avg[0] + 480
                            e_y = -el_var_avg[1] + 360 - 100 

                            #PID
                            yaw_pid = -0.8*(PD(e_x, e_x_old, del_t,K_P,K_D))
                            throttle_pid = 1.4*(PD(e_y, e_y_old, del_t,K_P,K_D))

                            # avoid -7 to 7
                            if yaw_pid > 2:
                                yaw_pid = yaw_pid + 6
                            elif yaw_pid < -2:
                                yaw_pid = yaw_pid - 6
                            else:
                                yaw_pid = 0

                            if throttle_pid > 2:
                                throttle_pid = throttle_pid + 7
                            elif throttle_pid < -2:
                                throttle_pid = throttle_pid - 7
                            else:
                                throttle_pid = 0

                            e_x_old = e_x
                            e_y_old = e_y

                            drone.send_rc_control(0, 0, int(throttle_pid), int(yaw_pid))

                            # if error is stable for 0.5s -> go to the next process 
                            if e_x < 100 and e_x > -100 and e_y < 100 and e_y >-100:
                                iter_center = iter_center + 1
                            else:
                                iter_center = 0
                            
                            if iter_center == 15: # iter 15 -> 0.3s
                                mission_process = 2
                                iter_center = 0

                            iter[1] = 20

                ## process 1.2
                elif mission_process == 2: # control drone to hover in front of the flag in distance 50cm
                    iter[2] = iter[2] + 1
                    iter[1] = 0

                    if iter[2] == 3:
                        e_x = 0
                        e_y = 0
                        e_dist = 0
                        e_x_old = 0
                        e_y_old = 0
                        e_dist_old = 0

                    if iter[2] == 10:

                        del_t = 0.1

                        K_P = 0.05
                        K_D = 0.03

                        #error save
                        e_x = -el_var_avg_d[0] + 480
                        e_y = -el_var_avg_d[1] + 360 - 100 
                        e_dist = -el_var_avg_d[2] + 120 #110 #140

                        #Boost
                        if e_x > 200 or e_x < -200:
                            roll_boost = 2
                        elif e_x > 150 and e_x <= 200:
                            roll_boost = 1.5
                        elif e_x < -150 and e_x >= -200:
                            roll_boost = 1.5
                        else:
                            roll_boost = 1
                        
                        if e_y > 200 or e_y < -200:
                            throttle_boost = 1.4
                        elif e_y > 150 and e_x <= 200:
                            throttle_boost = 1.2
                        elif e_y < -150 and e_x >= -200:
                            throttle_boost = 1.2
                        else:
                            throttle_boost = 1

                        if e_dist > 75 or e_dist < -75:
                            pitch_boost = 2
                        elif e_dist > 35 and e_dist <= 75:
                            pitch_boost = 1.5
                        elif e_dist < -35 and e_dist >= -75:
                            pitch_boost = 1.5
                        else:
                            pitch_boost = 1

                        #PID
                        roll_pid = -0.7*roll_boost*(PD(e_x, e_x_old, del_t, K_P-0.015, K_D))
                        throttle_pid = 1.5*throttle_boost*(PD(e_y, e_y_old, del_t, K_P, K_D))
                        pitch_pid = 1.8*pitch_boost*(PD(e_dist, e_dist_old, del_t, K_P, K_D))

                        # avoid -7 to 7
                        if roll_pid > 2:
                            roll_pid = roll_pid + 6
                        elif roll_pid < -2:
                            roll_pid = roll_pid - 6
                        elif roll_pid > 50:   # max speed limit
                            roll_pid = 50
                        elif roll_pid < -50:
                            roll_pid = -50 
                        else:
                            roll_pid = 0

                        if throttle_pid > 2:
                            throttle_pid = throttle_pid + 6
                        elif throttle_pid < -2:
                            throttle_pid = throttle_pid - 6
                        else:
                            throttle_pid = 0

                        if pitch_pid > 1.5:
                            pitch_pid = pitch_pid + 8
                        elif throttle_pid < -1.5:
                            pitch_pid = pitch_pid - 8
                        elif pitch_pid > 70:   # max speed limit
                            pitch_pid = 70
                        elif pitch_pid < -70:
                            pitch_pid = -70     
                        else:
                            pitch_pid = 0

                        e_x_old = e_x
                        e_y_old = e_y
                        e_dist_old = e_dist

                        drone.send_rc_control(int(roll_pid), int(pitch_pid), int(throttle_pid), 0)

                        # if error is stable for 1s -> go to next process 
                        if e_x < 120 and e_x > -120 and e_y < 120 and e_y >-120 and e_dist < 25 and e_dist > -25 :
                            iter_center = iter_center + 1
                        else:
                            iter_center = 0
                        
                        if iter_center == 10: # iter 20 -> 0.2s
                            mission_process = 3
                            iter_center = 0

                        iter[2] = 5

                ## process 1.3
                elif mission_process == 3: # detect circle and print color, capture picture 

                    iter[3] = iter[3] + 1
                    iter[2] = 0
                    drone.send_rc_control(0, 0, 0, 0)
                    print(color)

                    if iter[3] == 10:
                        mission_process = 4 

                ## process 1.4
                elif mission_process == 4:  # down & read handwriting & do QR mission % up
                    
                    iter[3] = 0

                    iter_yolo = iter_yolo + 1

                    # YOLO for number
                    if iter_yolo == 5 and not t_digit.is_alive():
                        if(not t_model.is_alive()):
                            t_model=Thread(target=model34,args=(model_num,image_gray2))
                            t_model.start()
                        if(modelinfo is not None):
                            number_handwriting = modelinfo
                            number_bound_info = number_handwriting.pandas().xyxy[0]

                        #number point raw save
                            if number_bound_info.shape[0] >= 1:
                                for j in range(0,number_bound_info.shape[0]):
                                    number_h_filt = int(number_bound_info.iloc[j,3])-int(number_bound_info.iloc[j,1])
                                    number_w_filt = int(number_bound_info.iloc[j,2])-int(number_bound_info.iloc[j,0])

                                    if number_h_filt <= 200 and number_w_filt <= 200 and number_bound_info.iloc[0,5] != 0:
                                        for i in range(0,4):
                                            number_var[i] = int(number_bound_info.iloc[0,i])
                                        number_var[4] = number_bound_info.iloc[0,5]
                                        number_detect = True
                                        break
                            else:
                                number_detect = False

                        iter_yolo = 0

                        number_handwriting_x = int((number_var[2]+number_var[0])/2)
                        number_handwriting_y = int((number_var[3]+number_var[1])/2)
                        number_handwriting_height = int(number_var[3]-number_var[1])

                        if number_detect == True:
                            cv2.rectangle(image_out, (int(number_var[0]),int(number_var[1])),(int(number_var[2]),int(number_var[3])),(255,0,255),2)
                            cv2.putText(image_out,str(number_var[4]),(int(number_var[0]),int(number_var[1])-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2)
                            digit = number_var[4]
                            print(digit)

                            if digit == digit_old:
                                iter_digit = iter_digit + 1
                            else:
                                iter_digit = 0

                            print(iter_digit)

                            digit_old = digit 
                
                            if(not t_digit.is_alive()) and iter_digit == 5:
                                drone.send_rc_control(0, 0, 0, 0)
                                is_before_mission = True

                                if digit == 9:
                                    mission9(image_out)
                                else:
                                    t_digit = Thread(target = digit_mission,args=(digit,))
                                    t_digit.start()

                                cv2.putText(image_out,'digit : ' + str(digit),(750,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,238,0),2)
                                
                                filt_iter = 0
                                filt_iter_d = 0
                                marker = False
                                iter_yolo = 0
                                pic_save = True
                                mission_state = mission_state + 1
                                mission_process = 0

                        else :
                            if altitude < 40:   # digit is 40 ~ 60cm 
                                drone.send_rc_control(0,0,10,0) 
                                is_up = True
                            elif altitude >=70:
                                is_up = False
                                drone.send_rc_control(0,0,-10,0) 
                            else:
                                if(is_up):                     
                                    drone.send_rc_control(0,0,10,0) 
                                else:
                                    drone.send_rc_control(0,0,-10,0)

            ################################### mission state 3 : Tracking ################################################
            if mission_state == 3 and not t_digit.is_alive():
                
                iter_digit = 0
                iter_yolo = iter_yolo + 1

                # YOLO for KAUmarker photo
                if iter_yolo == 5:
                    if(not t_model.is_alive()):
                        t_model=Thread(target=model34,args=(model_3,image))
                        t_model.start()
                    if(modelinfo is not None):
                        KAUmarker = modelinfo
                        KAU_bound_info = KAUmarker.pandas().xyxy[0]

                    #KAU_marker point raw save
                        if KAU_bound_info.shape[0] >= 1:
                            KAU_h_filt = int(KAU_bound_info.iloc[0,3])-int(KAU_bound_info.iloc[0,1])
                            KAU_w_filt = int(KAU_bound_info.iloc[0,2])-int(KAU_bound_info.iloc[0,0])

                            if KAU_bound_info.iloc[0,4] > 0.0 and KAU_h_filt <= 600 and KAU_w_filt <= 600:
                                for i in range(0,4):
                                    KAUmarker_var_raw[i] = int(KAU_bound_info.iloc[0,i])
                                KAUmarker_detect = True
                        else:
                            KAUmarker_detect = False
                    iter_yolo = 0

                #simpleKalman filter
                if filt_iter == 0:
                    KAUmarker_var_avg = KAUmarker_var_raw
                    filt_iter = 1

                KAUmarker_var_avg, P0_3 = simpleKalman(KAUmarker_var_avg, P0_3, KAUmarker_var_raw, 9)

                KAUmarker_x = int((KAUmarker_var_avg[2]+KAUmarker_var_avg[0])/2)
                KAUmarker_y = int((KAUmarker_var_avg[3]+KAUmarker_var_avg[1])/2)
                KAUmarker_height = int(KAUmarker_var_avg[3]-KAUmarker_var_avg[1])

                if KAUmarker_detect == True:
                    cv2.rectangle(image_out, (int(KAUmarker_var_avg[0]),int(KAUmarker_var_avg[1])),(int(KAUmarker_var_avg[2]),int(KAUmarker_var_avg[3])),(255,0,255),2)
                    cv2.circle(image_out,(KAUmarker_x,KAUmarker_y),2,(255,0,255),2)
                    cv2.putText(image_out,'Size : '+str(KAUmarker_height),(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,238,0),2)
                    cv2.putText(image_out,'KAU marker',(int(KAUmarker_var_avg[0]),int(KAUmarker_var_avg[1])-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)

                    if pic_save == True and mission_process == 2:
                        cv2.imwrite("C:/Users/chang/Desktop/detect_image/KAUmarker_Detected.png",image_out)
                        pic_save = False

                ## process 3.0
                if mission_process == 0 and not t_mission_up.is_alive():
                    iter[0] = iter[0] + 1
                    iter[1] = 0 
                    iter[3] = 0
                    if iter[0] == 10:
                        drone.send_rc_control(0, 0, 0, -35) 
                        iter[0] = 0
                    if KAUmarker_detect == True:
                        mission_process = 1

                ## process 3.1
                if mission_process == 1:

                    if KAUmarker_detect == False:
                        iter[0] = iter[0] + 1

                        if iter[0] == 3: # if no marker detected for 5*3 frames -> go back to mission process 0 
                            mission_process = 0

                    else:

                        iter[1] = iter[1] + 1
                        iter[0] = 0

                        if iter[1] == 10:
                            drone.send_rc_control(0, 0, 0, -15) # first inertia
                            e_x = 0
                            e_y = 0
                            e_x_old = 0
                            e_y_old = 0

                        if iter[1] == 19:
                            drone.send_rc_control(0, 0, 0, 0) 

                        if iter[1] == 25: # 5 iter -> 0.1s

                            del_t = 0.1

                            K_P = 0.05
                            K_D = 0.03

                            #error save
                            e_x = -KAUmarker_x + 480 
                            #e_y = -KAUmarker_y + 360 - 120 
                            e_y = (KAUmarker_y - KAUmarker_height/2) - 150

                            #PID
                            yaw_pid = -0.8*(PD(e_x, e_x_old, del_t,K_P,K_D))
                            throttle_pid = 1.4*(PD(e_y, e_y_old, del_t,K_P,K_D))*-1

                            # avoid -7 to 7
                            if yaw_pid > 2:
                                yaw_pid = yaw_pid + 6
                            elif yaw_pid < -2:
                                yaw_pid = yaw_pid - 6
                            else:
                                yaw_pid = 0

                            if throttle_pid > 2:
                                throttle_pid = throttle_pid + 7
                            elif throttle_pid < -2:
                                throttle_pid = throttle_pid - 7
                            else:
                                throttle_pid = 0

                            e_x_old = e_x
                            e_y_old = e_y

                            drone.send_rc_control(0, 0, int(throttle_pid), int(yaw_pid))

                            # if error is stable for 0.5s -> go to the next process 
                            if e_x < 100 and e_x > -100 and e_y < 100 and e_y > -100:
                                iter_center = iter_center + 1
                            else:
                                iter_center = 0
                            
                            if iter_center == 15: # iter 20 -> 0.3s
                                mission_process = 2
                                iter_center = 0

                            iter[1] = 20
                
                ## process 3.2
                elif mission_process == 2: 
                    iter[2] = iter[2] + 1
                    iter[1] = 0

                    yaw_state_c = drone.get_yaw()

                    if iter[2] == 3:
                        e_x = 0
                        e_y = 0
                        e_yaw = 0
                        e_dist = 0
                        e_x_old = 0
                        e_y_old = 0
                        e_yaw_old = 0
                        e_dist_old = 0


                    if iter[2] == 10:

                        del_t = 0.1

                        K_P = 0.05
                        K_D = 0.03

                        #error save
                        e_x = -KAUmarker_x + 480
                        #e_y = -KAUmarker_y + 360 - 120
                        e_y = (KAUmarker_y - KAUmarker_height/2) - 150
                        e_dist = -KAUmarker_height + 160 #120 #200
                        e_yaw = - yaw_state_c - 135

                        #Boost
                        if e_x > 200 or e_x < -200:
                            roll_boost = 1.5
                        elif e_x > 150 and e_x <= 200:
                            roll_boost = 1.2
                        elif e_x < -150 and e_x >= -200:
                            roll_boost = 1.2
                        else:
                            roll_boost = 1
                        
                        if e_y > 200 or e_y < -200:
                            throttle_boost = 2
                        elif e_y > 150 and e_x <= 200:
                            throttle_boost = 1.5
                        elif e_y < -150 and e_x >= -200:
                            throttle_boost = 1.5
                        else:
                            throttle_boost = 1

                        if e_dist > 75 or e_dist < -75:
                            pitch_boost = 4
                        elif e_dist > 35 and e_dist <= 75:
                            pitch_boost = 2
                        elif e_dist < -35 and e_dist >= -75:
                            pitch_boost = 2
                        else:
                            pitch_boost = 1

                        #PID
                        roll_pid = -0.7*roll_boost*(PD(e_x, e_x_old, del_t, K_P-0.01, K_D))
                        throttle_pid = 1.0*throttle_boost*(PD(e_y, e_y_old, del_t, K_P, K_D))*-1
                        pitch_pid = 2.0*pitch_boost*(PD(e_dist, e_dist_old, del_t, K_P, K_D))
                        yaw_pid = 4.5*(PD(e_yaw, e_yaw_old, del_t, K_P-0.02, K_D))

                        # avoid -7 to 7
                        if roll_pid > 1.5:
                            roll_pid = roll_pid + 6
                        elif roll_pid < -1.5:
                            roll_pid = roll_pid - 6
                        elif roll_pid > 25:   # max speed limit
                            roll_pid = 25
                        elif roll_pid < -25:
                            roll_pid = -25 
                        else:
                            roll_pid = 0

                        if throttle_pid > 2:
                            throttle_pid = throttle_pid + 6
                        elif throttle_pid < -2:
                            throttle_pid = throttle_pid - 6
                        else:
                            throttle_pid = 0

                        if pitch_pid > 1:
                            pitch_pid = pitch_pid + 10
                        elif throttle_pid < -1:
                            pitch_pid = pitch_pid - 10
                        elif pitch_pid > 40:   # max speed limit
                            pitch_pid = 40
                        elif pitch_pid < -40:
                            pitch_pid = -40     
                        else:
                            pitch_pid = 0

                        if yaw_pid > 0.3:
                                yaw_pid = yaw_pid + 6
                        elif yaw_pid < -0.3:
                            yaw_pid = yaw_pid - 6
                        elif yaw_pid > 13:   # max speed limit
                            yaw_pid = 13
                        elif yaw_pid < -13:
                            yaw_pid = -13
                        else:
                            yaw_pid = 0

                        e_x_old = e_x
                        e_y_old = e_y
                        e_dist_old = e_dist
                        e_yaw_old = e_yaw

                        drone.send_rc_control(int(roll_pid), int(pitch_pid), int(throttle_pid), int(yaw_pid))

                        # if marker disappear more than 1 sec go to next state
                        if KAUmarker_detect == False:
                            iter_center = iter_center + 1
                        else:
                            iter_center = 0

                        if iter_center == 3:
                            drone.send_rc_control(0,0,0,0)
                        
                        if iter_center == 15: # iter 15*5 -> 1.5s
                            t_state35 = Thread(target=state_35)
                            t_state35.start() 
                            filt_iter = 0
                            iter_center = 0
                            iter_yolo = 0
                            pic_save = True
                            mission_state = 4
                            mission_process = 0

                        iter[2] = 5

        ################################### mission state 4 : F22 Detection and Landing ################################################
            if mission_state == 4 and not t_state35.is_alive():

                alt_state_c = drone.get_distance_tof()

                if alt_state_c > 65:
                    drone.send_rc_control(0, 0, -15, 0)

                elif alt_state_c <= 65 : 

                    iter_yolo = iter_yolo + 1

                    # YOLO for F22photo
                    if iter_yolo == 5:
                        if(not t_model.is_alive()):
                            t_model=Thread(target=model34,args=(model_4,image_gray2))
                            t_model.start()
                        if(modelinfo is not None):
                            F22marker = modelinfo
                            F22_bound_info = F22marker.pandas().xyxy[0]

                            # f22_marker point raw save
                            if F22_bound_info.shape[0] >= 1:
                                for j in range(0,F22_bound_info.shape[0]):
                                    F22_h_filt = int(F22_bound_info.iloc[j,3])-int(F22_bound_info.iloc[j,1])
                                    F22_w_filt = int(F22_bound_info.iloc[j,2])-int(F22_bound_info.iloc[j,0])
                                    F22_y = int((F22_bound_info.iloc[j,1] + F22_bound_info.iloc[j,3])/2)

                                    if F22_bound_info.iloc[j,5] == 3 and F22_h_filt <= 450 and F22_w_filt <= 450 and F22_y >= 100 and F22_bound_info.iloc[j,4] > 0.2:                      
                                        for i in range(0,4):
                                            F22marker_var_raw[i] = int(F22_bound_info.iloc[j,i])
                                        F22marker_detect = True
                                        break
                            else:
                                F22marker_detect = False

                        iter_yolo = 0

                    #simpleKalman filter
                    if filt_iter == 0:
                        F22marker_var_avg = F22marker_var_raw
                        filt_iter = 1

                    F22marker_var_avg, P0_4 = simpleKalman(F22marker_var_avg, P0_4, F22marker_var_raw, 9)

                    F22marker_x = int((F22marker_var_avg[2]+F22marker_var_avg[0])/2)
                    F22marker_y = int((F22marker_var_avg[3]+F22marker_var_avg[1])/2)
                    F22marker_height = int(F22marker_var_avg[3]-F22marker_var_avg[1])

                    if F22marker_detect == True:
                        cv2.rectangle(image_out, (int(F22marker_var_avg[0]),int(F22marker_var_avg[1])),(int(F22marker_var_avg[2]),int(F22marker_var_avg[3])),(255,0,255),2)
                        cv2.circle(image_out,(F22marker_x,F22marker_y),2,(255,0,255),2)
                        cv2.putText(image_out,'Size : '+str(F22marker_height),(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,238,0),2)
                        cv2.putText(image_out,'F-22',(int(F22marker_var_avg[0]),int(F22marker_var_avg[1])-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)

                        if pic_save == True and mission_process == 2:
                            cv2.imwrite("C:/Users/chang/Desktop/detect_image/F22_Detected.png",image_out)
                            pic_save = False

                    ## process 4.0
                    if mission_process == 0 and not t_mission_up.is_alive():
                        iter[0] = iter[0] + 1
                        iter[1] = 0
                        iter[2] = 0 
                        iter[3] = 0
                        if iter[0] == 10:
                            drone.send_rc_control(0, 0, 0, -30) 
                            iter[0] = 0
                        if F22marker_detect == True:
                            mission_process = 1

                    ## process 4.1
                    if mission_process == 1:

                        if F22marker_detect == False:
                            iter[0] = iter[0] + 1

                            if iter[0] == 5: # if no marker detected for 5*5 frames -> go back to mission process 0 
                                mission_process = 0

                        else:
                            iter[1] = iter[1] + 1
                            iter[0] = 0

                            if iter[1] == 10:
                                drone.send_rc_control(0, 0, 0, 0)
                                e_x = 0
                                e_y = 0
                                e_x_old = 0
                                e_y_old = 0

                            if iter[1] == 19:
                                drone.send_rc_control(0, 0, 0, 0) 

                            if iter[1] == 25: # 5 iter -> 0.1s

                                del_t = 0.1

                                K_P = 0.05
                                K_D = 0.03

                                #error save
                                e_x = -F22marker_x + 480
                                e_y = (F22marker_y - F22marker_height/2) - 100

                                #PID
                                yaw_pid = -0.8*(PD(e_x, e_x_old, del_t,K_P,K_D))
                                throttle_pid = 1.4*(PD(e_y, e_y_old, del_t,K_P,K_D))*-1

                                # avoid -7 to 7
                                if yaw_pid > 2:
                                    yaw_pid = yaw_pid + 6
                                elif yaw_pid < -2:
                                    yaw_pid = yaw_pid - 6
                                else:
                                    yaw_pid = 0

                                if throttle_pid > 2:
                                    throttle_pid = throttle_pid + 5
                                elif throttle_pid < -2:
                                    throttle_pid = throttle_pid - 7
                                else:
                                    throttle_pid = 0

                                e_x_old = e_x
                                e_y_old = e_y

                                drone.send_rc_control(0, 0, int(throttle_pid), int(yaw_pid))

                                # if error is stable for 0.5s -> go to the next process 
                                if e_x < 120 and e_x > -120 and e_y < 120 and e_y >-120:
                                    iter_center = iter_center + 1
                                else:
                                    iter_center = 0
                                
                                if iter_center == 10: # iter 20 -> 0.2s
                                    mission_process = 2
                                    iter_center = 0

                                iter[1] = 20
                    
                    ## process 4.2
                    elif mission_process == 2: 
                        iter[2] = iter[2] + 1
                        iter[1] = 0

                        yaw_state_c = drone.get_yaw()

                        if iter[2] == 3:
                            e_x = 0
                            e_y = 0
                            e_yaw = 0
                            e_dist = 0
                            e_x_old = 0
                            e_y_old = 0
                            e_yaw_old = 0
                            e_dist_old = 0

                        if iter[2] == 10:

                            del_t = 0.1

                            K_P = 0.05
                            K_D = 0.03

                            #error save
                            e_x = -F22marker_x + 480 
                            e_y = (F22marker_y - F22marker_height/2) - 100
                            e_dist = -F22marker_height + 240 #160 #240
                            e_yaw = - yaw_state_c - 135

                            #Boost
                            if e_x > 200 or e_x < -200:
                                roll_boost = 1.4
                            elif e_x > 150 and e_x <= 200:
                                roll_boost = 1.2
                            elif e_x < -150 and e_x >= -200:
                                roll_boost = 1.2
                            else:
                                roll_boost = 1
                            
                            if e_y > 200 or e_y < -200:
                                throttle_boost = 1.4
                            elif e_y > 150 and e_x <= 200:
                                throttle_boost = 1.1
                            elif e_y < -150 and e_x >= -200:
                                throttle_boost = 1.1
                            else:
                                throttle_boost = 1

                            if e_dist > 75 or e_dist < -75:
                                pitch_boost = 3
                            elif e_dist > 35 and e_dist <= 75:
                                pitch_boost = 1.5
                            elif e_dist < -35 and e_dist >= -75:
                                pitch_boost = 1.5
                            else:
                                pitch_boost = 1

                            #PID
                            roll_pid = -0.9*roll_boost*(PD(e_x, e_x_old, del_t, K_P-0.01, K_D))
                            throttle_pid = 1.4*throttle_boost*(PD(e_y, e_y_old, del_t, K_P, K_D))*-1
                            pitch_pid = 2.5*pitch_boost*(PD(e_dist, e_dist_old, del_t, K_P, K_D))
                            yaw_pid = 4.5*(PD(e_yaw, e_yaw_old, del_t, K_P-0.02, K_D))

                            # avoid -7 to 7
                            if roll_pid > 1.5:
                                roll_pid = roll_pid + 6
                            elif roll_pid < -1.5:
                                roll_pid = roll_pid - 6
                            elif roll_pid > 50:   # max speed limit
                                roll_pid = 50
                            elif roll_pid < -50:
                                roll_pid = -50 
                            else:
                                roll_pid = 0

                            if throttle_pid > 2:
                                throttle_pid = throttle_pid + 4
                            elif throttle_pid < -2:
                                throttle_pid = throttle_pid - 6
                            else:
                                throttle_pid = 0

                            if pitch_pid > 1:
                                pitch_pid = pitch_pid + 10
                            elif throttle_pid < -1:
                                pitch_pid = pitch_pid - 10
                            elif pitch_pid > 70:   # max speed limit
                                pitch_pid = 70
                            elif pitch_pid < -70:
                                pitch_pid = -70     
                            else:
                                pitch_pid = 0

                            if yaw_pid > 0.3:
                                    yaw_pid = yaw_pid + 6
                            elif yaw_pid < -0.3:
                                yaw_pid = yaw_pid - 6
                            elif yaw_pid > 13:   # max speed limit
                                yaw_pid = 13
                            elif yaw_pid < -13:
                                yaw_pid = -13
                            else:
                                yaw_pid = 0

                            e_x_old = e_x
                            e_y_old = e_y
                            e_dist_old = e_dist
                            e_yaw_old = e_yaw

                            drone.send_rc_control(int(roll_pid), int(pitch_pid), int(throttle_pid), int(yaw_pid))

                            print(e_x,e_y,e_dist,e_yaw)

                            # if marker stable go to next step
                            if e_x < 100 and e_x > -100  and  e_yaw < 10 and e_yaw > -10:
                                iter_center = iter_center + 1
                            else:
                                iter_center = 0
                            
                            if iter_center == 20: # iter 20 -> 0.4s
                                mission_process = 3
                                iter_center = 0

                            iter[2] = 5

                    ## process 4.3
                    elif mission_process == 3:
                        
                        iter[3] = iter[3] + 1
                        iter[2] = 0

                        if iter[3] == 1:
                             drone.send_rc_control(0,0,0,0)

                        if iter[3] == 2:
                            cv2.imwrite("C:/Users/chang/Desktop/detect_image/F22_Detected2.png",image_out)

                        if iter[3] == 5:
                            #sleep(3)
                            drone.move_forward(160)
                            sleep(4)
                            drone.land()
                            cv2.destroyAllWindows()
                            drone.end()
            
        ####################################################### HUD ##################################################################
            #show center
            cv2.circle(image_out,(480, 360),2,(0, 238, 0),2)
            cv2.circle(image_out,(480, 360),13,(0, 238, 0),2)
            cv2.line(image_out,(420,360),(467,360),(0,238,0),2,8)
            cv2.line(image_out,(493,360),(540,360),(0,238,0),2,8)
            cv2.line(image_out,(420,360),(420,340),(0,238,0),2,8)
            cv2.line(image_out,(540,360),(540,340),(0,238,0),2,8)

            #Yaw indicator
            yaw_state = drone.get_yaw()
            if yaw_state < 0:
                yaw_state = yaw_state + 360

            cv2.line(image_out,(250,80),(710,80),(0,238,0),1,8)
            cv2.line(image_out,(480,95),(470,80),(0,238,0),2,8)
            cv2.line(image_out,(480,95),(490,80),(0,238,0),2,8)
            cv2.line(image_out,(470,80),(490,80),(0,238,0),2,8)
            # cv2.line(image_out,(470,80),(480,70),(0,238,0),2,8)
            # cv2.line(image_out,(480,70),(490,80),(0,238,0),2,8)

            yaw_tick_st = 480 - int(90*(yaw_state%5)/5)
            yaw_tick2_st = yaw_state - (yaw_state%5)
            yaw_ticks = [yaw_tick_st - 180, yaw_tick_st - 90, yaw_tick_st, yaw_tick_st + 90, yaw_tick_st + 180]
            yaw_ticks2 = [yaw_tick2_st - 10, yaw_tick2_st - 5, yaw_tick2_st, yaw_tick2_st + 5, yaw_tick2_st + 10]

            for i in range(0,5):
                if yaw_ticks2[i] < 0:
                    yaw_ticks2[i] = yaw_ticks2[i] + 360
                if yaw_ticks2[i] >= 360:
                    yaw_ticks2[i] = yaw_ticks2[i] - 360

            for i in range(0,5):
                if yaw_ticks[i] >= 250 and yaw_ticks[i] <= 710: 
                    cv2.line(image_out,(yaw_ticks[i],90),(yaw_ticks[i],75),(0,238,0),2,8)
                    cv2.line(image_out,(yaw_ticks[i]+45,90),(yaw_ticks[i]+45,75),(0,238,0),1,8)
                    if yaw_ticks2[i]%10 == 0:
                        cv2.putText(image_out,str(yaw_ticks2[i]),(yaw_ticks[i]-11,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,238,0),2)
                        cv2.line(image_out,(yaw_ticks[i],95),(yaw_ticks[i],70),(0,238,0),2,8)
            
            #Roll indicator
            roll_state = drone.get_roll()
            
            cv2.ellipse(image_out,(480, 360),(210,210),0,235,305,(0, 238, 0),1)
            cv2.line(image_out,(480,140),(480,160),(0,238,0),2,8)

            for i in range(-7,8):
                roll_tick1 = rotate_mat(0, 205, 5*i)
                roll_tick2 = rotate_mat(0, 215, 5*i)
                roll_tick3 = rotate_mat(0, 230, 5*i)
                cv2.line(image_out,(480 - int(roll_tick1[0]), 360 - int(roll_tick1[1])),(480 - int(roll_tick2[0]), 360 - int(roll_tick2[1])),(0,238,0),1,8)

                if i%2 == 0:
                    cv2.line(image_out,(480 - int(roll_tick1[0]), 360 - int(roll_tick1[1])),(480 - int(roll_tick2[0]), 360 - int(roll_tick2[1])),(0,238,0),2,8)
                    if i == 0:
                        tick_comp = 6
                    if i < 0:
                        tick_comp = 10
                    if i > 0:
                        tick_comp = 20
                    cv2.putText(image_out,str(-i*5),(480 - int(roll_tick3[0])-tick_comp, 360 - int(roll_tick3[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),2)
            
            roll_ind_m = rotate_mat(0,200,-roll_state)
            roll_ind_l = rotate_mat(-10,185,-roll_state)
            roll_ind_r = rotate_mat(10,185,-roll_state)
            roll_hor_l1 = rotate_mat(-50,0,-roll_state)
            roll_hor_l2 = rotate_mat(-275,0,-roll_state)
            roll_hor_r1 = rotate_mat(50,0,-roll_state)
            roll_hor_r2 = rotate_mat(275,0,-roll_state)
            cv2.line(image_out,(480 - int(roll_ind_m[0]), 360 - int(roll_ind_m[1])),(480 - int(roll_ind_l[0]), 360 - int(roll_ind_l[1])),(0,238,0),2,8)
            cv2.line(image_out,(480 - int(roll_ind_m[0]), 360 - int(roll_ind_m[1])),(480 - int(roll_ind_r[0]), 360 - int(roll_ind_r[1])),(0,238,0),2,8)
            cv2.line(image_out,(480 - int(roll_hor_l1[0]), 360 - int(roll_hor_l1[1])),(480 - int(roll_hor_l2[0]), 360 - int(roll_hor_l2[1])),(0,238,0),1,8)
            cv2.line(image_out,(480 - int(roll_hor_r1[0]), 360 - int(roll_hor_r1[1])),(480 - int(roll_hor_r2[0]), 360 - int(roll_hor_r2[1])),(0,238,0),1,8)

            # Pitch Indicator
            pitch_state = drone.get_pitch()

            pitch_tick_st = -int(100*(pitch_state%5)/5)
            pitch_tick2_st = pitch_state - (pitch_state%5)
            pitch_ticks = [pitch_tick_st - 300, pitch_tick_st - 200, pitch_tick_st - 100, pitch_tick_st, pitch_tick_st + 100, pitch_tick_st + 200, pitch_tick_st + 300]
            pitch_ticks2 = [pitch_tick2_st - 15, pitch_tick2_st - 10, pitch_tick2_st - 5, pitch_tick2_st, pitch_tick2_st + 5, pitch_tick2_st + 10, pitch_tick2_st + 15]

            for i in range(0,7):
                pitch_hor_l1 = rotate_mat(80, pitch_ticks[i], -roll_state)
                pitch_hor_l2 = rotate_mat(130, pitch_ticks[i], -roll_state)
                pitch_hor_l3 = rotate_mat(190, pitch_ticks[i], -roll_state)
                pitch_hor_r1 = rotate_mat(-80, pitch_ticks[i], -roll_state)
                pitch_hor_r2 = rotate_mat(-130, pitch_ticks[i], -roll_state)
                pitch_hor_r3 = rotate_mat(-150, pitch_ticks[i], -roll_state)

                if pitch_ticks[i] >= -250 and pitch_ticks[i] <= 250: 
                    cv2.line(image_out,(480 - int(pitch_hor_l1[0]), 360 - int(pitch_hor_l1[1])),(480 -  int(pitch_hor_l2[0]), 360 - int(pitch_hor_l2[1])),(0,238,0),2,8)
                    cv2.line(image_out,(480 - int(pitch_hor_r1[0]), 360 - int(pitch_hor_r1[1])),(480 -  int(pitch_hor_r2[0]), 360 - int(pitch_hor_r2[1])),(0,238,0),2,8)

                    for j in range(-2,3):
                        pitch_hor_m_l1 = rotate_mat(110, pitch_ticks[i] + 20*j, -roll_state)
                        pitch_hor_m_l2 = rotate_mat(120, pitch_ticks[i] + 20*j, -roll_state)
                        pitch_hor_m_r1 = rotate_mat(-110, pitch_ticks[i] + 20*j, -roll_state)
                        pitch_hor_m_r2 = rotate_mat(-120, pitch_ticks[i] + 20*j, -roll_state)
                        cv2.line(image_out,(480 - int(pitch_hor_m_l1[0]), 360 - int(pitch_hor_m_l1[1])),(480 -  int(pitch_hor_m_l2[0]), 360 - int(pitch_hor_m_l2[1])),(0,238,0),1,8)
                        cv2.line(image_out,(480 - int(pitch_hor_m_r1[0]), 360 - int(pitch_hor_m_r1[1])),(480 -  int(pitch_hor_m_r2[0]), 360 - int(pitch_hor_m_r2[1])),(0,238,0),1,8)

                    if pitch_ticks2[i]%5 == 0:
                        cv2.putText(image_out,str(-pitch_ticks2[i]),(480 - int(pitch_hor_l3[0]), 360 - int(pitch_hor_l3[1])),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,238,0),2)
                        cv2.putText(image_out,str(-pitch_ticks2[i]),(480 - int(pitch_hor_r3[0]), 360 - int(pitch_hor_r3[1])),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,238,0),2)
            
            # Battery Indicator
            battery_state = drone.get_battery()

            cv2.rectangle(image_out, (40,150),(60,600),(0,238,0),1)
            cv2.rectangle(image_out, (40,150 + int(4.5*(100-battery_state))),(60,600),(0,180,0),-1)
            cv2.putText(image_out,str(battery_state),(40,142),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),2)
            cv2.putText(image_out,'BAT',(37,125),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),2)

            for i in range(0,11):
                cv2.line(image_out,(30,150+45*i),(60,150+45*i),(0,238,0),1,8)

            #Speed Indicator
            x_speed = drone.get_speed_x()
            y_speed = drone.get_speed_y()
            z_speed = drone.get_speed_z()

            speed_tot = int(math.sqrt(math.pow(10*x_speed,2)+math.pow(10*y_speed,2)+math.pow(10*z_speed,2))) # speed in mm/s

            cv2.putText(image_out,str(speed_tot),(175, 365),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),2)
            cv2.line(image_out,(160,360),(170,350),(0,238,0),2,8)
            cv2.line(image_out,(160,360),(170,370),(0,238,0),2,8)

            cv2.putText(image_out,'SPD',(137,125),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),2)
            cv2.putText(image_out,'[mm/s]',(125,142),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,238,0),1)
            cv2.line(image_out,(135,150),(160,150),(0,238,0),2,8)
            cv2.line(image_out,(135,570),(160,570),(0,238,0),2,8)
            cv2.line(image_out,(160,150),(160,570),(0,238,0),2,8)

            speed_tick_st = - int(70*(speed_tot%10)/10)
            speed_tick2_st = speed_tot - (speed_tot%10)
            speed_ticks = [speed_tick_st - 210,speed_tick_st - 140,speed_tick_st - 70,speed_tick_st, speed_tick_st + 70, speed_tick_st + 140, speed_tick_st + 210]
            speed_ticks2 = [speed_tick2_st - 30, speed_tick2_st - 20, speed_tick2_st - 10, speed_tick2_st, speed_tick2_st + 10, speed_tick2_st + 20, speed_tick2_st + 30]

            for i in range(0,7):
                if speed_ticks[i] >= -210 and speed_ticks[i] <= 210: 
                    cv2.line(image_out,(140, 360 - speed_ticks[i]),(160, 360 - speed_ticks[i]),(0,238,0),2,8)
                    if speed_ticks[i]+35 >= -210 and speed_ticks[i]+35 <= 210:
                        cv2.line(image_out,(150, 360 - (speed_ticks[i]+35)),(160, 360 - (speed_ticks[i]+35)),(0,238,0),1,8)
                    if speed_ticks2[i]%10 == 0:
                        cv2.putText(image_out,str(speed_ticks2[i]),(100, 360 - speed_ticks[i]+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),1)
                    
            #Altimeter
            alt_state = drone.get_distance_tof()

            cv2.putText(image_out,str(alt_state),(750, 365),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),2)
            cv2.line(image_out,(800,360),(790,350),(0,238,0),2,8)
            cv2.line(image_out,(800,360),(790,370),(0,238,0),2,8)

            cv2.putText(image_out,'ALT',(797,125),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),2)
            cv2.putText(image_out,'[cm]',(798,142),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,238,0),1)
            cv2.line(image_out,(800,150),(825,150),(0,238,0),2,8)
            cv2.line(image_out,(800,570),(825,570),(0,238,0),2,8)
            cv2.line(image_out,(800,150),(800,570),(0,238,0),2,8)

            alt_tick_st = - int(70*(alt_state%10)/10)
            alt_tick2_st = alt_state - (alt_state%10)
            alt_ticks = [alt_tick_st - 210, alt_tick_st - 140, alt_tick_st - 70, alt_tick_st, alt_tick_st + 70, alt_tick_st + 140, alt_tick_st + 210]
            alt_ticks2 = [alt_tick2_st - 30, alt_tick2_st - 20, alt_tick2_st - 10, alt_tick2_st, alt_tick2_st + 10, alt_tick2_st + 20, alt_tick2_st + 30]

            for i in range(0,7):
                if alt_ticks[i] >= -210 and alt_ticks[i] <= 210: 
                    cv2.line(image_out,(800, 360 - alt_ticks[i]),(820, 360 - alt_ticks[i]),(0,238,0),2,8)
                    if alt_ticks[i]+35 >= -210 and alt_ticks[i]+35 <= 210:
                        cv2.line(image_out,(800, 360 - (alt_ticks[i]+35)),(810, 360 - (alt_ticks[i]+35)),(0,238,0),1,8)
                    if alt_ticks2[i]%10 == 0:
                        cv2.putText(image_out,str(alt_ticks2[i]),(830, 360 - alt_ticks[i]+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),1)

            # Vertical Speed Indicator
            throttle_state = -drone.get_speed_z()

            cv2.rectangle(image_out, (900,150),(920,570),(0,238,0),1)
            if throttle_state <= 15 and throttle_state >= -15:
                cv2.rectangle(image_out, (900,570 - int(70*(15 + throttle_state)/5)),(920,570),(0,180,0),-1)
            elif throttle_state >= 15:
                cv2.rectangle(image_out, (900,150),(920,570),(0,180,0),-1)
            elif throttle_state <= -15:
                cv2.rectangle(image_out, (900,570),(920,570),(0,180,0),-1)

            cv2.putText(image_out,str(throttle_state),(865, 575 - int(70*(15 + throttle_state)/5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),2)
            cv2.line(image_out,(900,570 - int(70*(15 + throttle_state)/5)),(890,560 - int(70*(15 + throttle_state)/5)),(0,238,0),2,8)
            cv2.line(image_out,(900,570 - int(70*(15 + throttle_state)/5)),(890,580 - int(70*(15 + throttle_state)/5)),(0,238,0),2,8)

            cv2.putText(image_out,'[cm/s]',(885,142),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,238,0),1)
            cv2.putText(image_out,'THR',(897,125),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,238,0),2)
            for i in range(0,7):
                cv2.line(image_out,(900,150+70*i),(930,150+70*i),(0,238,0),1,8)
                cv2.putText(image_out,str(15 - i*5),(930,155+70*i),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,238,0),1)

            # Mission Time
            flight_time = drone.get_flight_time()
            cv2.putText(image_out,'Flight Time : '+str(flight_time)+'sec',(700,710),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            # show mission state
            cv2.putText(image_out,'MISSION STATE : '+str(mission_state),(20,670),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(image_out,'MISSION PROCESS : '+str(mission_process),(20,710),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            # window 
            cv2.imshow('Archaeopteryx', image_out)
            #cv2.imshow('dilate', img_dilate)

            if(cv2.waitKey(1)>0):
                k = True
            
            if(k):
                cv2.destroyAllWindows() 
                drone.land()  
                drone.end()  
                break  

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)

    finally:
        #out.release()
        drone.end()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()