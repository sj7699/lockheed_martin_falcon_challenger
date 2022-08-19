import sys
import traceback
import tellopy
import av
import cv2
import numpy as np
import time
import os,sys
from threading import Thread
from time import sleep

drone = tellopy.Tello()
def drone_state(roll,pitch,yaw,throttle): # roll,pitch,yaw,trottle -> -100 ~ 100 integer
    global drone
    print("roll "+str(roll))
    print("pitch "+str(pitch))
    print("yaw "+str(yaw))
    print("throttle "+str(throttle))
    
    if roll >= 0:
        drone.right(roll)
    elif roll < 0:
        drone.left(-roll)

    if pitch >= 0:
        drone.forward(pitch)
    elif pitch < 0:
        drone.backward(-pitch)

    if yaw >= 0:
        drone.clockwise(yaw)
    elif yaw < 0:
        drone.counter_clockwise(-yaw)

    if throttle >= 0:
        drone.up(throttle)
    elif throttle < 0:
        drone.down(-throttle)
    
def handler(event, sender, data, **args):
    drone = sender
    #if event is drone.EVENT_FLIGHT_DATA:
        #print(data)

def main():
    global drone
    # variables initialization
    iter = 0
    iter2 = 0
    iter3 = 0
    yaw_s = 0
    throttle_s = -10
    mode = 0
    GB_detect = 0
    n = 0

    try:
        
        drone.connect()
        drone.wait_for_connection(60.0)
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
        drone.connect()
        drone.wait_for_connection(60.0)
        print("take off!")
        drone.takeoff()
        print("take off finished")

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
                ##container = av.open('13.mp4')
            except av.AVError as ave:
                print(ave)
                print('retry...')

        # skip first 500 frames
        frame_skip = 300 #500
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)

                #variables
                square_detect = 0
                QR_detect = 0
                color = -1

                #brighten
                image = image + 0

                #blur
                image_blur = cv2.GaussianBlur(image,(0,0),3,3)

                #rgb -> gray
                image_gray = cv2.cvtColor(image_blur, cv2.COLOR_RGB2GRAY)

                #canny
                canny_min = 50
                canny_max = 90
                image_canny = cv2.Canny(image_blur, canny_min, canny_max)

                #dilation
                kernel = np.ones((3,3), int)
                image_dilate = cv2.dilate(image_canny, kernel, iterations=1)

                #QR
                detector=cv2.QRCodeDetector()
                decodedText, points, _ = detector.detectAndDecode(image)
                print(decodedText)

                if points is None:
                    QR_detect = 0
                    # contour detection(square)
                    contours,hier = cv2.findContours(image_dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                    for cnt in contours:
                        x,y,w,h = cv2.boundingRect(cnt)
                        area = cv2.contourArea(cnt)             

                        if area > 10000:
                            square_detect = 1
                            #cv2.drawContours(image,[cnt],0,(255,0,255),4)
                            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),3)

                            # color detection point
                            cx = int(x+(w/2))
                            cy = int(y+(h/2))

                            detect_point = np.array([[cx,cx+20,cx+20,cx-20,cx-20], [cy,cy+20,cy-20,cy+20,cy-20]])
                            cv2.circle(image,(cx,cy),5,(255,255,255),2)

                            if cx + 20 >= 720:
                                detect_point[0,1] = cx
                                detect_point[0,2] = cx
                            if cx - 20 <= 0:
                                detect_point[0,3] = cx
                                detect_point[0,4] = cx
                            if cy + 20 >= 960:
                                detect_point[1,1] = cy
                                detect_point[1,3] = cy
                            if cy - 20 <= 0:
                                detect_point[1,2] = cy
                                detect_point[1,4] = cy
                            
                            sum_color = [0,0,0]
                            detect_color = [0,0,0]      
                
                            # color save
                            for i in range(0,5):
                                cp = image_blur[detect_point[1,i],detect_point[0,i]]
                                sum_color = sum_color + cp

                                detect_color = sum_color/5
                                
                                dom_index = detect_color.argmax()

                                if dom_index == 0:
                                    color = 0
                                    GB_detect = 1
                                    cv2.putText(image,"BLUE",(cx,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)
                                elif dom_index == 1:
                                    color = 1
                                    GB_detect = 1
                                    cv2.putText(image,"GREEN",(cx,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,128,0),2)
                                elif dom_index == 2:
                                    color = 2
                                    cv2.putText(image,"RED",(cx,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2) 

                elif len(decodedText) > 0:
                    QR_detect = 1
                    print(decodedText)     

                # drone control
                if square_detect == 1 and color == 2:
                    mode = 1 # hovering mode

                if QR_detect == 1 and GB_detect == 1:
                    mode = 4 # land mode
                
                if mode == 1:
                    trottle_s = 0 # 3sec hover
                    iter = iter + 1

                    if iter >= 10: #90
                        mode = 2 # rotation mode

                if mode == 2: # after 3sec hovering rotation start 
                    yaw_s = -10
                    throttle_s = 0
                    iter2 = iter2 + 1 
                    iter3 = 0

                    if iter2 >= 300: #300
                       mode = 3
                       n = n + 1

                if mode == 3: # rotation end descend bit and rotate again 
                    yaw_s = 0

                    if (n//5)%2 == 0:
                        throttle_s = -10
                    elif (n//5)%2 == 1:
                        throttle_s = 10

                    iter3 = iter3 + 1
                    iter2 = 0

                    if iter3 >= 100: #100
                        mode = 2

                if mode == 4: # land mode
                    drone.land()
                    print("Land")
                    cv2.destroyAllWindows()
                    break
                
                roll = 0
                pitch = 0
                yaw = yaw_s
                throttle = throttle_s

                drone_state(roll,pitch,yaw,throttle)
                print("mode "+str(mode))
                # print("square detect "+str(square_detect))
                # print("GB detect "+str(GB_detect))
                # print("QR detect "+str(QR_detect))
                # print("iter2 "+str(iter2))
                # print("iter3 "+str(iter3))
                # print("n"+str(n))

                cv2.imshow('Original', image)
                #cv2.imshow('Canny', image_dilate)
                cv2.waitKey(1)

                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)
                    
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()