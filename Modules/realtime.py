import cv2
import threading
import time
from recap import URI, CfgNode as CN
from queue import Queue

from DetectBoard import find_corners
def process_frame(frame):
    # Your processing logic here
    # For demonstration purposes, let's just display the frame
    cv2.imshow('Processed Frame', frame)
    cv2.waitKey(1)

def processing_thread_worker(cfg, frame, result_queue):
    corners_result = find_corners(cfg, frame)
    print("Burasi")
    print(corners_result)
    result_queue.put(corners_result.tolist())  

def video_capture():
    
    ip_camera_url = f"http://192.168.1.64/video"

    cap = cv2.VideoCapture(ip_camera_url)  # Replace 'your_video.mp4' with your video file
    width = 1200
    height = 675
    cfg = CN.load_yaml_with_base("C:\\Users\\Monster\\Desktop\\Graduation Project1\\ChessPlayerHelperDesktop\\configuration\\corner_detection.yaml")
    result_queue = Queue()

    cv2.namedWindow("Frames", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frames", width, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1200, 675))

        cv2.imshow('Original Frame', frame)

        # Check if the 'a' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            
            print('pressed')
            # Create a new thread for processing the frame
            processing_thread = threading.Thread(target=processing_thread_worker, args=(cfg,frame, result_queue))
            processing_thread.start()
            # processing_thread.join()

            # # Retrieve the result from the queue
            # corners_result = result_queue.get()
            # corners_result = np.array(corners_result)  # Convert the list back to NumPy array
            # print("Result from processing:", corners_result)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_capture()