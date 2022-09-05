import sys
import traceback
import tellopy
import av
import cv2  # for avoidance of pylint error
import numpy
import time
import pyzbar.pyzbar as pyzbar

def main():
    drone = tellopy.Tello()

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        fourcc=cv2.VideoWriter_fourcc(*'DIVX')
        out=cv2.VideoWriter('ouput_drone_4.avi',fourcc,30,(960,720))
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

                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue

                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                cv2.imshow('Original', image)
                cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                qrs=pyzbar.decode(image)
                if(len(qrs)==0): print("noqr")
                if(len(qrs)==1): print("single ",qrs[0].data.decode('utf-8'))
                if(len(qrs)>1): print("multiple ",qrs[0].data.decode('utf-8'))
                p=cv2.waitKey(1)
                if(p>0):
                    out.release()
                    break
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
