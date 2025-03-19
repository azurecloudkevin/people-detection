# import the necessary packages
import numpy as np
import cv2
import os
import datetime
import threading
import shutil
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Get video input details from environment variables:
VIDEO_INPUT = os.getenv("VIDEO_INPUT", None)
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "./outputs")
OUTPUT_FPS = int(os.getenv("OUTPUT_FPS", "30"))
OUTPUT_WIDTH = int(os.getenv("OUTPUT_WIDTH", "640"))
OUTPUT_HEIGHT = int(os.getenv("OUTPUT_HEIGHT", "480"))

def start_output_video():
    folder_name = "./" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #create folder
    os.makedirs(folder_name, exist_ok=True)
    # create file
    if not os.path.exists(folder_name):
        os.makedirs(os.path.dirname(folder_name), exist_ok=True)

    return folder_name

   
def end_output_video(folder):
    # create a thread to convert images to video
    video_create_thread = threading.Thread(target=create_video_from_images, args=(folder,))
    video_create_thread.start()
    # wait for the thread to finish
    video_create_thread.join()
    # release the video writer object   
    print("Video creation thread finished.")
    # remove the folder
    shutil.rmtree(folder)
    print(f"Folder {folder} removed.")

def create_video_from_images(folder):
    # read all files in the folder
    files = os.listdir(folder)
    # sort files by name
    files.sort()
    # create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use a valid codec like 'mp4v'
    fps = 30  # Set the desired frames per second
    width = 640  # Set the desired width of the video
    height = 480  # Set the desired height of the video
    file_name = f"./outputs/{folder}.mp4"
    file_count = len(files)
    video_writer = cv2.VideoWriter(file_name, fourcc, fps, (width, height))  # Ensure valid parameters

    for file in files:
        # read the image
        img = cv2.imread(os.path.join(folder, file))
        # resize the image to the desired dimensions
        img = cv2.resize(img, (width, height))
        # write the image to the video writer object
        video_writer.write(img)
    # release the video writer object  
    video_writer.release()
    print(f"Video created at {file_name} with {file_count} images.")

def main():
    
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()
    # open webcam video stream
    # cap = cv2.VideoCapture(0)


    n = 0
    os.makedirs('./outputs', exist_ok=True)
    base_path = "./outputs"
    ext = 'jpg'
    recording = False
    zero_detections = 0
    folder = "./"
    pic_index = 0

    while(True):
        # Capture frame-by-frame
        if VIDEO_INPUT is not None:
            cap = cv2.VideoCapture(VIDEO_INPUT)
        else:
            cap = cv2.VideoCapture(0)

        ret, frame = cap.read()

        # resizing for faster detection
        frame = cv2.resize(frame, (640, 480))
        # using a greyscale picture, also for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB),
                            (0, 255, 0), 2)

        # Write the output video 
        # out.write(frame)
        # out.write(frame.astype('uint8'))
        if boxes.size > 0:
            zero_detections = 0
            if not recording:
                recording = True
                folder = start_output_video()
            
            print(f"Recording in {folder}")
            # create a four digit version of pic_index
            pic_index_str = str(pic_index).zfill(4)
            cv2.imwrite(f"./{folder}/image-{pic_index_str}.jpg", frame)
            pic_index += 1
            print(f"Recording... {pic_index}")
        else:
            if recording:
                zero_detections += 1
                if zero_detections > 10:
                    recording = False
                    end_output_video(folder)
                    pic_index = 0
                else:
                    cv2.imwrite(f"{folder}/image-{pic_index}.jpg", frame)
                    pic_index += 1
                    print(f"Recording... {pic_index}")

        # Display the resulting frame
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    if recording:
        end_output_video(folder)
    cap.release()
    # finally, close the window
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main")
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
    