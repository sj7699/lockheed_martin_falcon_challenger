import sys
import traceback
import tellopy
import av
import cv2 as cv2  # for avoidance of pylint error
import numpy as np
import time


def main():
    drone = tellopy.Tello()

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        # skip first 300 frames
        frame_skip = 300
        while True:
            for frame in container.decode(video=0):

                start_time = time.time()
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                
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
                    if(contour_area<1000 or extend<0.6):
                        continue
                    detect_rect=contour
                    detect_color=np.argmax(image[rect_y+int(rect_h/2)][rect_x+int(rect_w/2)])
                    break
                if(detect_rect is not None):                        
                    cv2.drawContours(image,detect_rect,-1,(0,0,255),4)
                cv2.imshow('Original', image)
                cv2.imshow('Canny', cv2.Canny(image, 100, 200))
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
