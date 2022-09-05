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

def PD(e,e_old,del_t,K_P,K_D):

    out = e * K_P + (e - e_old) * K_D / del_t

    return out

def dist_measure(r_pix):
    r = 22 # circle true diameter [cm]
    vert_ang = 31.6

    x = (r*320)/((r_pix)*math.tan(vert_ang*math.pi/180))

    return x

def down(time):
    global drone
    print("Down")
    drone.send_rc_control(0,0,10,0)
    sleep(time)
    drone.send_rc_control(0,0,0,0)
    print("Down Finish")

def up(dist):
    global drone
    print("Up")
    drone.move_up(dist)
    sleep(3)
    drone.send_rc_control(0,0,0,0)
    print("Up Finish")

def mission1(dist):
    global drone
    print("Up " + str(dist) + "cm")
    drone.move_up(dist)
    sleep(3)
    print("Down " + str(dist) + "cm")
    drone.move_down(dist)
    sleep(3)
    print("Up & Down Finish")

def mission2():
    global drone
    print("Flip Forward")
    sleep(3)
    drone.move_back(20) # 앞으로 flip 하므로 다시 뒤로 복귀해야 할 듯 싶습니다.
    sleep(3)
    drone.flip_forward()
    sleep(3)
    print("Flip Forward Finish")

def mission3(dist):
    global drone
    print("Down " + str(dist) + "cm")
    drone.move_down(dist)
    sleep(3)
    print("Up " + str(dist) + "cm")
    drone.move_up(dist)
    sleep(3)
    print("Down & Up Finish")

def mission4():
    global drone
    print("Flip Left")
    sleep(3)
    drone.flip_left()
    sleep(3)
    drone.move_right(20) # 왼쪽으로 flip 하므로 다시 오른쪽으로 복귀해야 할 듯 싶습니다.
    sleep(3)
    print("Flip Left Finish") 

def mission5():
    global drone
    print("Rotate CW 360 Degree")
    drone.rotate_clockwise(360)
    sleep(3)
    print("Rotate Finish")

calc = 0    # QR calculation's result (1~5)
def qr_mission():
    global drone,calc
    print("mission " + str(calc) +" start!")
    if calc == 1:
        #t_mission1 = Thread(target=mission1, args = 30)
        #t_mission1.start()
        mission1(30)
    elif calc == 2:
        #t_mission2 = Thread(target=mission2)
        #t_mission2.start()
        mission2()
    elif calc == 3:
        #t_mission3 = Thread(target=mission3, args = 30)
        #t_mission3.start()
        mission3(30)
    elif calc == 4:
        #t_mission4 = Thread(target=mission4)
        #t_mission4.start()
        mission4()
    elif calc ==5:
        #t_mission5 = Thread(target=mission5)
        #t_mission5.start()
        mission5()

def hover():
    global drone
    drone.send_rc_control(0,0,0,0)
    sleep(7)
    print("Hovering Finish after QR Detected")

def main():
    
    #variables definition
    global drone,calc
    is_before_mission=False
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
    P0 = np.array([6,6,6,6,6])
    P0_d = np.array([6,6,6,6,6])
    detect_color = None
    color_detect_desire = None
    mission_state = -1  # which flag(color) to detect (-1: QR, 0: GREEN, 1: RED, 2: BLUE)
    mission_process = -1   # drone control process at the mission state
    is_up = False
    
    try:
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('output103.avi',fourcc,30,(960,720))

        drone.connect()
        drone.takeoff()
        print("take off!")
        sleep(5)
        print("take off finished")
        height_takeoff = drone.get_height()      # altitude right after take off
        print("Now altitude : " + str(height_takeoff))
        bat = drone.get_battery()
        print("Now battery : " + str(bat))
        drone.move_back(20)

        # video start
        drone.streamon()
        frame_read = drone.get_frame_read()

        # thread definition
        t_hover = Thread(target=hover)
        t_qr = Thread(target=qr_mission)
        t_mission_up= Thread(target=up,args=(30,))
        while True:
            k = False
            marker = False

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

            #el_exist = False
            area_max = 1

            for cnt in contours:
                
                # ellipse detection
                (el_cx, el_cy),(el_h,el_w),angle = cv2.fitEllipse(cnt) #타원찾는거?
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

                if fit >= 0.95 and area > 3500 and detect_color == color_detect_desire:
                    marker = True
                    
                    #save maximum area ellipse
                    if (area - area_max)/area_max >= 0.1:
                        area_max = area 
                        raw_data_d = np.array([el_cx,el_cy,el_h/2,el_w/2,angle])
                        #el_exist = True

                    raw_data = np.array([el_cx,el_cy,el_h/2,el_w/2,angle])
                    
                    #simpleKalman filter
                    if filt_iter == 0:
                        el_var_avg = raw_data
                        filt_iter = 1

                    el_var_avg, P0 = simpleKalman(el_var_avg, P0, raw_data, 144)   
 
                    #show center of the flag
                    el_center = (int(el_var_avg[0]),int(el_var_avg[1]))

                    cv2.circle(image_out,el_center,2,(0,0,0),3)
                    cv2.ellipse(image_out,(int(el_cx),int(el_cy)),(int(el_h/2),int(el_w/2)),angle,0,360,(255,255,255),2)

                    #color str save
                    if detect_color == 0:
                        color = 'BLUE'           
                    elif detect_color == 1:
                        color = 'GREEN'               
                    elif detect_color == 2:
                        color = 'RED'                   

            if marker == True:             
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
            altitude = drone.get_height()
        
            if mission_state == -1: # down&up and read QR for hovering 5seconds
                
                # QR detect 1 right after take-off
                detectqr=pyzbar.decode(image)
                if(len(detectqr)!=0):
                    print(detectqr[0].data.decode('utf-8'))
                    print("Hovering Start!")
                    if(not t_hover.is_alive()): t_hover.start()
                    mission_state = mission_state + 1
                    mission_process = 0  
            
            if mission_process ==0 and is_before_mission and not t_qr.is_alive() and not t_mission_up.is_alive():
                    t_mission_up=Thread(target=up,args=(30,))
                    t_mission_up.start() 
                    is_before_mission=False
            
            if mission_process == 0 and not t_hover.is_alive() and not t_qr.is_alive() and not t_mission_up.is_alive(): # rotate counter-clockwise until find the marker
                iter_0 = iter_0 + 1
                iter_1 = 0 
                iter_3 = 0
                if iter_0 == 10:
                    drone.send_rc_control(0, 0, 0, -40) 
                    iter_0 = 0
                if marker:
                    mission_process = 1

            elif mission_process == 1: # PID control make center of the ellipse be center of the image frame
                
                if marker == False:
                    iter_0 = iter_0 + 1

                    if iter_0 == 8: # if no marker detected for 8 frames -> go back to mission process 0 
                        mission_process = 0

                else:
                    iter_1 = iter_1 + 1
                    iter_0 = 0

                    if iter_1 == 10:
                        drone.send_rc_control(0, 0, 0, -15) # first inertia
                        e_x = 0
                        e_y = 0
                        e_x_old = 0
                        e_y_old = 0

                    if iter_1 == 19:
                        drone.send_rc_control(0, 0, 0, 0) 

                    if iter_1 == 25: # 5 iter -> 0.1s

                        del_t = 0.1

                        K_P = 0.05
                        K_D = 0.03

                        #error save
                        e_x = -el_var_avg[0] + 480
                        e_y = -el_var_avg[1] + 360 

                        #PID
                        yaw_pid = -0.95*(PD(e_x, e_x_old, del_t,K_P,K_D))
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
                        if e_x < 60 and e_x > -60 and e_y < 60 and e_y >-60:
                            iter_center = iter_center + 1
                        else:
                            iter_center = 0
                        
                        if iter_center == 25: # iter 50 -> 0.5s
                            mission_process = 2
                            iter_center = 0

                        iter_1 = 20

            elif mission_process == 2: # control drone to hover in front of the flag in distance 50cm
                iter_2 = iter_2 + 1
                iter_1 = 0

                if iter_2 == 3:
                    e_x = 0
                    e_y = 0
                    e_dist = 0
                    e_x_old = 0
                    e_y_old = 0
                    e_dist_old = 0

                if iter_2 == 10:

                    del_t = 0.1

                    K_P = 0.05
                    K_D = 0.03

                    #error save
                    e_x = -el_var_avg_d[0] + 480
                    e_y = -el_var_avg_d[1] + 360
                    e_dist = - dist_d + 45

                    #PID
                    roll_pid = -0.8*(PD(e_x, e_x_old, del_t, K_P-0.01, K_D))
                    throttle_pid = 1.4*(PD(e_y, e_y_old, del_t, K_P, K_D))
                    pitch_pid = -1.8*(PD(e_dist, e_dist_old, del_t, K_P, K_D))

                    #print(roll_pid,pitch_pid,throttle_pid)

                    # avoid -7 to 7
                    if roll_pid > 2:
                        roll_pid = roll_pid + 6
                    elif roll_pid < -2:
                        roll_pid = roll_pid - 6
                    elif roll_pid > 20:   # max speed limit
                        roll_pid = 20
                    elif roll_pid < -20:
                        roll_pid = -20 
                    else:
                        roll_pid = 0

                    if throttle_pid > 2:
                        throttle_pid = throttle_pid + 6
                    elif throttle_pid < -2:
                        throttle_pid = throttle_pid - 6
                    else:
                        throttle_pid = 0

                    if pitch_pid > 2:
                        pitch_pid = pitch_pid + 8
                    elif throttle_pid < -2:
                        pitch_pid = pitch_pid - 8
                    elif pitch_pid > 30:   # max speed limit
                        pitch_pid = 30
                    elif pitch_pid < -30:
                        pitch_pid = -30     
                    else:
                        pitch_pid = 0

                    e_x_old = e_x
                    e_y_old = e_y
                    e_dist_old = e_dist

                    drone.send_rc_control(int(roll_pid), int(pitch_pid), int(throttle_pid), 0)

                    # if error is stable for 1s -> go to next process 
                    if e_x < 70 and e_x > -70 and e_y < 70 and e_y >-70 and e_dist < 40 and e_dist > -40 :
                        iter_center = iter_center + 1
                    else:
                        iter_center = 0
                    
                    if iter_center == 25: # iter 25 -> 0.5s
                        mission_process = 3
                        iter_center = 0

                    iter_2 = 5

            elif mission_process == 3: # detect circle and print color 

                iter_3 = iter_3 + 1
                iter_2 = 0
                drone.send_rc_control(0, 0, 0, 0)
                print(color)

                if iter_3 == 10:
                    mission_process = 4 

            elif mission_process == 4:  # down & read QR & do QR mission % up
                
                # QR detect 2nd
                detectqrs=pyzbar.decode(image)
                now_str=""
                if(len(detectqrs)!=0): now_str=detectqrs[0].data.decode('utf-8')
                if(len(now_str)!=0):
                    print(now_str)
                    calc = eval(now_str)
                    print(calc)
                    if(not t_qr.is_alive()):
                        drone.send_rc_control(0, 0, 0, 0)
                        is_before_mission=True
                        t_qr=Thread(target=qr_mission)
                        t_qr.start()
                    mission_state = mission_state + 1
                    mission_process = 0

                else :
                    if altitude < 40:   # QR is 40 ~ 60cm 
                        drone.send_rc_control(0,0,10,0) 
                        is_up=True
                    elif altitude >=70:
                        is_up=False
                        drone.send_rc_control(0,0,-10,0) 
                    else:
                        if(is_up):                     
                            drone.send_rc_control(0,0,10,0) 
                        else:
                            drone.send_rc_control(0,0,-10,0)


            if mission_state == 3 and not t_qr.is_alive() : # Mission End
                drone.land()
                drone.end()  
                
            #show center
            cv2.circle(image_out,(480, 360),15,(255, 0, 255),2)
            cv2.line(image_out,(470,360),(490,360),(255,0,255),2,8)
            cv2.line(image_out,(480,350),(480,370),(255,0,255),2,8)

            #show mission state
            cv2.putText(image_out,'MISSION STATE : '+str(mission_state),(20,640),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)
            cv2.putText(image_out,'MISSION PROCESS : '+str(mission_process),(20,700),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)

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
