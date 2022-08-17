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
mission_name=["","finding BLUE RECTANGLE","finding GREEN OR RED RECTANGLE","finding QR"]
def down(dist):
    print("Down "+str(dist))
    sleep(dist/10*2)
    print("DOWN FINISH")

def up(dist):
    print("UP "+str(dist))
    sleep(dist/10*2)
    print("UP FINISH")

def stop(t):
    print("stop")
    sleep(t)
    print("stop FINISH")

def rotate(angle):
    print("rotate "+str(angle))
    sleep(3)
    print("rotate FINISH")

def downandrotate(dist,angle):
    down(dist)
    rotate(angle)

def upandrotate(dist,angle):
    up(dist)
    rotate(angle)

def mission(altitude,is_up,mv_dist,mv_angle,t_up,t_down,mission_cnt,mission_state):
    if(is_up):
        if(not t_up.is_alive()):
            print("now mission is",mission_name[mission_state])
            if(altitude>=90):
                altitude-=10
                print("now altitude: "+str(altitude))
                if(mission_state==1): t_down=Thread(target=down,args=(mv_dist,))
                else: t_down=Thread(target=downandrotate,args=(mv_dist,mv_angle))
                t_down.start()
                is_up=False
                mission_cnt+=1
            else:
                altitude+=10
                print("now altitude: "+str(altitude))
                if(mission_state==1): t_up=Thread(target=up,args=(mv_dist,))
                else: t_up=Thread(target=upandrotate,args=(mv_dist,mv_angle))
                t_up.start()
                mission_cnt+=1
    else:
        if(not t_down.is_alive()):
            print("now mission is",mission_name[mission_state])
            if(altitude<=20):
                altitude+=10
                print("now altitude: "+str(altitude))
                if(mission_state==1): t_up=Thread(target=up,args=(mv_dist,))
                else: t_up=Thread(target=upandrotate,args=(mv_dist,mv_angle))
                t_up.start()
                is_up=True
                mission_cnt+=1
            else:
                altitude-=10
                print("now altitude: "+str(altitude))
                if(mission_state==1): t_down=Thread(target=down,args=(mv_dist,))
                else: t_down=Thread(target=downandrotate,args=(mv_dist,mv_angle))
                t_down.start()
                mission_cnt+=1 
    return altitude,is_up,t_up,t_down,mission_cnt

def main():
    #drone = tellopy.Tello()
    try:
    #     drone.connect()
    #     drone.wait_for_connection(60.0)
    #     drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
        print("take off!")
    #     drone.takeoff()
        sleep(5)
        print("take off finished")
        t_up=Thread(target=up,args=(10,))
        t_down = Thread(target=down,args=(50,))
        t_downandrotate=Thread(target=downandrotate,args=(10,400))
        t_upandrotate=Thread(target=upandrotate,args=(10,400))
        t_stop=Thread(target=stop,args=(4,))
        retry = 3
        container = None
        mission_state=1
        down(40)
        altitude=80
        mission1_cnt=0
        mission2_cnt=0
        mission3_cnt=0
        mv_dist=10
        mv_angle=400
        is_up=False
        fourcc=cv2.VideoWriter_fourcc(*'DIVX')
        out=cv2.VideoWriter('output2.avi',fourcc,30,(960,720))
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(r"C:\Users\sj\Documents\Tello_prac\test1.mp4")
            except av.AVError as ave:
                print(ave)
                print('retry...')
        frame_skip = 0
        while True:
            k=False
            for frame in container.decode(video=0):
                start_time = time.time()
                image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                detector=cv2.QRCodeDetector()
                decodedText, points, _ = detector.detectAndDecode(image)  
                img2 = cv2.Canny(image, 180, 180)   
                img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
                contours,hier = cv2.findContours(img2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                detect_rect=None
                detect_color=3
                for contour in contours:
                    rect_x,rect_y,rect_w,rect_h=cv2.boundingRect(contour)
                    contour_area=cv2.contourArea(contour)
                    if(np.max(image[rect_y+int(rect_h/2)][rect_x+int(rect_w/2)])<80):
                        continue
                    extend=float(contour_area)/(rect_w*rect_h)
                    if(rect_w<150 or rect_h<150 or extend<0.7):
                        continue
                    detect_rect=contour
                    detect_color=np.argmax(image[rect_y+int(rect_h/2)][rect_x+int(rect_w/2)])
                    break
                
                if(mission_state==1):
                    altitude,is_up,t_up,t_down,mission1_cnt=mission(altitude,is_up,mv_dist,mv_angle,t_up,t_down,mission1_cnt,mission_state)
                    if(detect_rect is not None and detect_color==0 and points is None): 
                        mission_state=2
                        print(colour[detect_color])
                        t_stop.start()
                if(mission_state==2 and not t_stop.is_alive()):
                    altitude,is_up,t_upandrotate,t_downandrotate,mission2_cnt=mission(altitude,is_up,mv_dist,mv_angle,t_upandrotate,t_downandrotate,mission2_cnt,mission_state)
                    if(detect_rect is not None and detect_color!=0 and points is None): 
                        mission_state=3
                        print(colour[detect_color])
                        t_stop=Thread(target=stop,args=(0.5,))
                        t_stop.start()
                if(mission_state==3 and not t_stop.is_alive()):
                    altitude,is_up,t_upandrotate,t_downandrotate,mission2_cnt=mission(altitude,is_up,mv_dist,mv_angle,t_upandrotate,t_downandrotate,mission2_cnt,mission_state)
                    if(points is not None):
                        print(decodedText)
                        print("land")
                        k=True
                        break
                if(t_stop.is_alive()):
                    if(detect_rect is not None):                        
                        cv2.drawContours(image,detect_rect,-1,(0,0,255),4)
                out.write(image)
                cv2.imshow('Original', image)
                cv2.imshow('canny',img2)
                if(cv2.waitKey(1)>0):
                    k=True
                    break
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)
            if(k):
                out.release()
                sleep(3)
                cv2.destroyAllWindows() 
                # drone.land()
                # drone.quit()    
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
