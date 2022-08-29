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

drone = Tello()

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

def PID(e,e_old,d_old,del_t,K_P,K_D,K_I):

    out = e * K_P + (e - e_old) * K_D / del_t + K_I * ((e + e_old)*del_t*0.5 + d_old) 

    return out

def dist_measure(r_pix):
    r = 15 # circle true radius [cm]
    vert_ang = 31.6

    x = (r*320)/((r_pix)*math.tan(vert_ang*math.pi/180))

    return x

def main():
    #variables
    global drone
    iter_0 = 0
    iter_1 = 0
    iter_2 = 0
    iter_3 = 0
    iter_center = 0
    filt_iter = 0
    filt_iter_d = 0
    e_x = 0
    e_y = 0
    e_dist = 0
    e_x_old = 0
    e_y_old = 0
    e_dist_old = 0
    d_x_old = 0
    d_y_old = 0
    d_dist_old = 0
    P0 = np.array([6,6,6,6,6])
    P0_d = np.array([6,6,6,6,6])
    detect_color = None
    color_detect_desire = None
    marker = False
    mission_state = 0  # which flag(color) to detect
    mission_process = 0   # drone control process at the mission state

    try:
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('output103.avi',fourcc,30,(960,720))

        drone.connect()
        drone.takeoff()
        print("take off!")
        sleep(5)
        print("take off finished")

        # video start
        drone.streamon()
        frame_read = drone.get_frame_read()
        
        while True:
            k = False

            image = frame_read.frame
            image_out = image.copy()

            #blur
            image_blur = cv2.GaussianBlur(image,(0,0),3,3)

            #gray
            image_gray = cv2.cvtColor(image_blur, cv2.COLOR_RGB2GRAY)

            #Canny  
            img_canny = cv2.Canny(image_blur, 70, 100)

            #Dilate    
            kernel=np.ones((3,3),int)
            img_dilate=cv2.dilate(img_canny,kernel,iterations=2) 

            #contour Detection
            contours,hier = cv2.findContours(img_dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            #mission state
            if mission_state == 0:   # first detect green
                color_detect_desire = 1
            elif mission_state == 1:   # second detect red
                color_detect_desire = 2
            elif mission_state == 2:    # last detect blue
                color_detect_desire = 0

            el_exist = False
            area_max = 1

            for cnt in contours:
                
                # ellipse detection
                (el_cx, el_cy),(el_h,el_w),angle = cv2.fitEllipse(cnt)
                area = cv2.contourArea(cnt)
                area_el = math.pi *el_h*el_w*0.25
                
                # avoid division 0
                if area_el == 0:
                    area_el = 100000
                    
                fit = float(area/area_el)

                # avoid center out of frame
                if el_cx >= 960 or el_cx < 0 or el_cy >= 720 or el_cy < 0:
                    continue

                #color detection / detect_color => 0 : Blue, 1 : Green , 2 : Red
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
                for i in range(0,5):
                    cp = image_blur[int(detect_point[1,i]),int(detect_point[0,i])]
                    sum_color = sum_color + cp

                v_color = sum_color/5
                detect_color = v_color.argmax()

                if fit >= 0.95 and area > 3000 and detect_color == color_detect_desire:
                    marker = True
                    
                    #save maximum area ellipse
                    if (area - area_max)/area_max >= 0.1:
                        area_max = area 
                        raw_data_d = np.array([el_cx,el_cy,el_h/2,el_w/2,angle])
                        el_exist = True

                    raw_data = np.array([el_cx,el_cy,el_h/2,el_w/2,angle])
                    
                    #simpleKalman filter
                    if filt_iter == 0:
                        el_var_avg = raw_data
                        filt_iter = 1

                    el_var_avg, P0 = simpleKalman(el_var_avg, P0, raw_data, 144)   
 
                    #show center of flag
                    el_center = (int(el_var_avg[0]),int(el_var_avg[1]))

                    cv2.circle(image_out,el_center,2,(0,0,0),3)
                    cv2.ellipse(image_out,(int(el_cx),int(el_cy)),(int(el_h/2),int(el_w/2)),angle,0,360,(0,0,0),2)

                    #color str save
                    if detect_color == 0:
                        color = 'BLUE'
                    elif detect_color == 1:
                        color = 'GREEN'
                    elif detect_color == 2:
                        color = 'RED'
                else:
                    marker = False

            if el_exist == True:
                
                #simpleKalman filter
                if filt_iter_d == 0:
                    el_var_avg_d = raw_data_d
                    filt_iter_d = 1

                el_var_avg_d, P0_d = simpleKalman(el_var_avg_d, P0_d, raw_data_d,9)   

                #show ellipse 
                el_center_d = (int(el_var_avg_d[0]),int(el_var_avg_d[1]))
                el_hw_d = (int(el_var_avg_d[2]),int(el_var_avg_d[3]))
                angle_t_d = int(el_var_avg_d[4])

                dist_d = int(dist_measure(el_var_avg_d[2]))

                cv2.ellipse(image_out,el_center_d,el_hw_d,angle_t_d,0,360,(255,0 ,255),3)
                cv2.circle(image_out,el_center_d,2,(255,0,255),2)
                cv2.putText(image_out,'long radius : '+str(int(el_var_avg_d[2])),(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),1)
                cv2.putText(image_out,color,(int(el_var_avg_d[0]),int(el_var_avg_d[1]+el_var_avg_d[2])),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)

            ######## drone control ############

            ## hovering mission after takeoff ##
            #   detect QR and hover for 5sec   #
            #            after                 #
            #    -> mission_state = 0          #
            #    -> mission_process = 0        #
            ####################################
            if mission_process == 0: # rotate counter-clockwise until find the marker
                iter_0 = iter_0 + 1  # 지금은 marker 찾기 까지 무작정 돌기 이지만 미션 수행을 위해 360 돌고 뒤로 빠지고로 고쳐야 할듯 함
                if iter_0 == 10:
                    drone.send_rc_control(0, 0, 0, -25) 
                    iter_0 = 0
                
                if marker:
                    mission_process = 1

            elif mission_process == 1: # PID control center of the ellipse be center of the image frame
                iter_1 = iter_1 + 1

                if iter_1 == 10:
                    drone.send_rc_control(0, 0, 0, -15) # first inertia
                    e_x = 0
                    e_y = 0
                    e_x_old = 0
                    e_y_old = 0
                    d_x_old = 0
                    d_y_old = 0

                if iter_1 == 19:
                    drone.send_rc_control(0, 0, 0, 0) 

                if iter_1 == 25: # 5 iter -> 0.1s

                    del_t = 0.1

                    K_P = 0.05
                    K_D = 0.03
                    K_I = 0

                    #error save
                    e_x = -el_var_avg[0] + 480
                    e_y = -el_var_avg[1] + 360 

                    #PID
                    yaw_pid = -0.95*(PID(e_x, e_x_old, d_x_old, del_t,K_P,K_D,K_I))
                    throttle_pid = 1.3*(PID(e_y, e_y_old, d_y_old, del_t,K_P,K_D,K_I))

                    # avoid -5 to 5
                    if yaw_pid > 2:
                        yaw_pid = yaw_pid + 6
                    elif yaw_pid < -2:
                        yaw_pid = yaw_pid - 6
                    else:
                        yaw_pid = 0

                    if throttle_pid > 2:
                        throttle_pid = throttle_pid + 6
                    elif throttle_pid < -2:
                        throttle_pid = throttle_pid - 6
                    else:
                        throttle_pid = 0

                    e_x_old = e_x
                    e_y_old = e_y

                    d_x_old = (e_x + e_x_old)*del_t*0.5 + d_x_old
                    d_y_old = (e_y + e_y_old)*del_t*0.5 + d_y_old

                    drone.send_rc_control(0, 0, int(throttle_pid), int(yaw_pid))

                    # if error is stable for 1s -> go to next process 
                    if yaw_pid == 0 and throttle_pid == 0:
                        iter_center = iter_center + 1
                    else:
                        iter_center = 0
                    
                    if iter_center == 50: # iter 50 -> 1s
                        mission_process = 2
                        iter_center = 0

                    iter_1 = 20

            elif mission_process == 2: # control drone to hover in front of the flag in distance 50cm
                iter_2 = iter_2 + 1

                if iter_2 == 3:
                    e_x = 0
                    e_y = 0
                    e_dist = 0
                    e_x_old = 0
                    e_y_old = 0
                    e_dist_old = 0
                    d_x_old = 0
                    d_y_old = 0
                    d_dist_old = 0

                if iter_2 == 10:

                    del_t = 0.1

                    K_P = 0.05
                    K_D = 0.03
                    K_I = 0

                    K_P_2 = 0.05 
                    K_D_2 = 0.03
                    K_I_2 = 0

                    #error save
                    e_x = -el_var_avg_d[0] + 480
                    e_y = -el_var_avg_d[1] + 360
                    e_dist = - dist_d + 70 

                    #PID
                    roll_pid = -(PID(e_x, e_x_old, d_x_old, del_t, K_P_2, K_D_2, K_I_2))
                    throttle_pid = 1.3*(PID(e_y, e_y_old, d_y_old, del_t, K_P, K_D, K_I))
                    pitch_pid = -(PID(e_dist, e_dist_old, d_dist_old, del_t, K_P_2, K_D_2, K_I_2))

                    #print(roll_pid,pitch_pid,throttle_pid)

                    # avoid -5 to 5
                    if roll_pid > 2:
                        roll_pid = roll_pid + 5
                    elif roll_pid < -2:
                        roll_pid = roll_pid - 5
                    elif roll_pid > 20:   # max speed limit
                        roll_pid = 20
                    elif roll_pid < -20:
                        roll_pid = -20 
                    else:
                        roll_pid = 0

                    if throttle_pid > 2:
                        throttle_pid = throttle_pid + 5
                    elif throttle_pid < -2:
                        throttle_pid = throttle_pid - 5
                    else:
                        throttle_pid = 0

                    if pitch_pid > 2:
                        pitch_pid = pitch_pid + 7
                    elif throttle_pid < -2:
                        pitch_pid = pitch_pid - 7
                    elif pitch_pid > 20:   # max speed limit
                        pitch_pid = 20
                    elif pitch_pid < -20:
                        pitch_pid = -20     
                    else:
                        pitch_pid = 0

                    e_x_old = e_x
                    e_y_old = e_y
                    e_dist_old = e_dist

                    d_x_old = (e_x + e_x_old)*del_t*0.5 + d_x_old
                    d_y_old = (e_y + e_y_old)*del_t*0.5 + d_y_old
                    d_dist_old = (e_dist + e_dist_old)*del_t*0.5 + d_dist_old

                    drone.send_rc_control(int(roll_pid), int(pitch_pid), int(throttle_pid), 0)
                    #print(e_x, e_y, e_dist)

                    # if error is stable for 1s -> go to next process 
                    if e_x < 50 and e_x > -50 and e_y < 50 and e_y >-50 and e_dist < 40 and e_dist > -40 :
                        iter_center = iter_center + 1
                    else:
                        iter_center = 0
                    
                    if iter_center == 50: # iter 50 -> 1s
                        mission_process = 3
                        iter_center = 0

                    iter_2 = 5

            elif mission_process == 3: # detect circle and print color 

                iter_3 = iter_3 + 1

                if iter_3 == 10:

                    drone.send_rc_control(0, 0, 0, 0)
                    iter_3 = 0

                    if mission_state == 0:
                        print('GREEN')
                    elif mission_state == 1:
                        print('RED')
                    elif mission_state == 2:
                        print('BLUE')

            #elif mission_process == 4: # read QR and do QR mission
            ## after mission process 4 -> mission_state + 1 and mission process = 0
                 
            #show center
            cv2.circle(image_out,(480, 360),15, (255, 0, 255),2 )
            cv2.line(image_out,(470,360),(490,360),(255,0,255), 2, 8)
            cv2.line(image_out,(480,350),(480,370),(255,0,255), 2, 8)

            #show mission state
            cv2.putText(image_out,'MISSION STATE : '+str(mission_state),(20,640),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),1)
            cv2.putText(image_out,'MISSION PROCESS : '+str(mission_process),(20,700),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),1)

            #window 
            cv2.imshow('Original', image_out)
            #cv2.imshow('threshold', img_dilate)

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
        drone.end()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
