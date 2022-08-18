import sys
import traceback
import tellopy
import av
import cv2  # for avoidance of pylint error
import numpy as np
import time
import os,sys
from threading import Thread
from time import sleep
dir=os.getcwd()
#sys.path.append(dir+r"\mission")
colour=["BLUE","GREEN","RED"]
mission_name=["","find BLUE RECTANGLE","find GREEN OR RED RECTANGLE","find QR"]
drone = tellopy.Tello()
altitude=12
def down(dist):
    global drone
    print("Down "+str(dist))
    drone.down(dist)
    sleep(5)
    print("DOWN FINISH")

def up(dist):
    global drone
    print("UP "+str(dist))
    drone.up(dist)
    sleep(5)
    print("UP FINISH")

def stop(t):
    global drone
    print("stop")
    drone.send_packet_data('stop')
    sleep(t)
    print("stop FINISH")

def stop2(t):
    global drone
    print("stop")
    drone.set_pitch=0
    drone.set_roll=0
    drone.set_throttle=0
    sleep(t)
    print("stop FINISH")

def rotate(angle):
    global drone
    print("rotate "+str(angle))
    drone.clockwise(angle)
    sleep(8)
    print("rotate FINISH")

def downandrotate(dist,angle):
    down(dist)
    rotate(angle)

def upandrotate(dist,angle):
    up(dist)
    rotate(angle)

def mission(is_up,mv_dist,mv_angle,t_up,t_down,mission_state):
    global altitude
    if(is_up):
        if(not t_up.is_alive()):
            print("now mission : ",mission_name[mission_state])
            if(altitude>=9):
                print("now altitude: "+str(altitude))
                if(mission_state==1): t_down=Thread(target=down,args=(mv_dist,))
                else: t_down=Thread(target=downandrotate,args=(mv_dist,mv_angle))
                t_down.start()
                is_up=False
            else:
                print("now altitude: "+str(altitude))
                if(mission_state==1): t_up=Thread(target=up,args=(mv_dist,))
                else: t_up=Thread(target=upandrotate,args=(mv_dist,mv_angle))
                t_up.start()
    else:
        if(not t_down.is_alive()):
            print("now mission : ",mission_name[mission_state])
            if(altitude<=2):
                print("now altitude: "+str(altitude))
                if(mission_state==1): t_up=Thread(target=up,args=(mv_dist,))
                else: t_up=Thread(target=upandrotate,args=(mv_dist,mv_angle))
                t_up.start()
                is_up=True
            else:
                print("now altitude: "+str(altitude))
                if(mission_state==1): t_down=Thread(target=down,args=(mv_dist,))
                else: t_down=Thread(target=downandrotate,args=(mv_dist,mv_angle))
                t_down.start()
    return is_up,t_up,t_down

def handler(event, sender, data, **args):
    global altitude
    here_drone = sender
    if event is here_drone.EVENT_FLIGHT_DATA:
        altitude=data.height
        
def main():
    global drone
    try:
        fourcc=cv2.VideoWriter_fourcc(*'DIVX')
        out=cv2.VideoWriter('output101.avi',fourcc,30,(960,720))
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
        drone.connect()
        drone.wait_for_connection(60.0)
        print("take off!")
        drone.takeoff()
        print("take off finished")
        t_up=Thread(target=up,args=(10,))
        t_down = Thread(target=down,args=(50,))
        t_downandrotate=Thread(target=downandrotate,args=(10,400))
        t_upandrotate=Thread(target=upandrotate,args=(10,400))
        t_stop=Thread(target=stop,args=(4,))
        retry = 3
        container = None
        mission_state=1
        mv_dist=10
        mv_angle=400
        is_up=False
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')
        frame_skip = 500
        while True:
            k=False
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                detector=cv2.QRCodeDetector()
                decodedText, points, _ = detector.detectAndDecode(image)  
                img2 = cv2.Canny(image, 180, 180)    
                kernel=np.ones((3,3),int)
                img2=cv2.dilate(img2,kernel,iterations=1) 
                #_,img_thresh=cv2.threshold(img2,127,255,cv2.THRESH_BINARY_INV)
                contours,hier = cv2.findContours(img2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                detect_rect=None
                detect_color=3
                for contour in contours:
                    rect_x,rect_y,rect_w,rect_h=cv2.boundingRect(contour)
                    contour_area=cv2.contourArea(contour)
                    extend=float(contour_area)/(rect_w*rect_h)
                    if(rect_w<15 or rect_h<15 or extend<0.6):
                        continue
                    detect_rect=contour
                    detect_color=np.argmax(image[rect_y+int(rect_h/2)][rect_x+int(rect_w/2)])
                    break
                if(mission_state==1):
                    is_up,t_up,t_down=mission(is_up,mv_dist,mv_angle,t_up,t_down,mission_state)
                    if(detect_rect is not None and detect_color==1 and points is None): 
                        mission_state=2
                        print(colour[detect_color])
                        t_stop.start()
                if(mission_state==2 and not t_stop.is_alive()):
                    is_up,t_upandrotate,t_downandrotate=mission(is_up,mv_dist,mv_angle,t_upandrotate,t_downandrotate,mission_state)
                    if(detect_rect is not None and detect_color!=1 and points is None): 
                        mission_state=3
                        print(colour[detect_color])
                        t_stop=Thread(target=stop,args=(0,5,))
                        t_stop.start()
                if(mission_state==3 and not t_stop.is_alive()):
                    is_up,t_upandrotate,t_downandrotate=mission(is_up,mv_dist,mv_angle,t_upandrotate,t_downandrotate,mission_state)
                    if(points is not None):
                        points_mean=np.mean(points,axis=1)[0]
                        print(points_mean)
                        if(points_mean[0]>250 and points_mean[0]<400 and points_mean[1]>200 and points_mean[1]<600):
                            print(decodedText)
                            print("land")
                            k=True
                            break
                if(t_stop.is_alive()):
                    if(detect_rect is not None):                        
                        cv2.drawContours(image,detect_rect,-1,(0,0,255),4)
                cv2.imshow('Original', image)
                cv2.imshow('canny',img2)
                if(cv2.waitKey(1)>0):
                    k=True
                    drone.land()
                    break
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)
            if(k):
                sleep(3)
                cv2.destroyAllWindows() 
                drone.quit()  
                drone.land()  
                break  

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        #drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
