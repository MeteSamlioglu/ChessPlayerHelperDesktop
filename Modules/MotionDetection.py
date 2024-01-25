import threading
import winsound
import cv2
import imutils 


#camera_url = "http://192.168.1.40:8080/video"
ip_camera_url = f"http://192.168.247.18:8080/video"

cap = cv2.VideoCapture(ip_camera_url)
#cap = cv2.VideoCapture(camera_url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


_, start_frame = cap.read()
start_frame = imutils.resize(start_frame, width = 500)
start_frame =  cv2.resize(start_frame, (1200, 675))
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)


alarm = False 
alarm_mode = False
alarm_counter = 0
is_motion_detected = False

    
while True:
    
    _, frame = cap.read()
    
    #frame = imutils.resize(frame, width = 500)
    
    frame = cv2.resize(frame, (1200, 675))

    if alarm_mode:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (5,5), 0)
        
        difference = cv2.absdiff(frame_bw, start_frame)
        threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
        start_frame = frame_bw


        #if threshold.sum() > 5000:  #If you smaller this value the sensitivity of movement detection will increase, opposite will decrease
        if threshold.sum() > 30:
            alarm_counter +=1
        else:
            if alarm_counter > 0:
                alarm_counter -= 1     
                 
        cv2.imshow("Cam",threshold)
    else:
        cv2.imshow("Cam", frame)
            
    if alarm_counter > 20:
            print("Motion is detected")
            is_motion_detected = True
            
    if alarm_counter == 0 and is_motion_detected:
        print("Captured")
        is_motion_detected = False
        cv2.imshow("Detecte motion", frame)
        cv2.waitKey(0)
    
    key_pressed = cv2.waitKey(1)
    if key_pressed == ord("t"):
        alarm_mode = not alarm_mode
        alarm_counter = 0
    if key_pressed == ord("q"):
        alarm_mode = False
        break

cap.release()
cv2.destroyAllWindows()
     
        
        