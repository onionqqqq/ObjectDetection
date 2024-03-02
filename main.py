import cv2
import math
import numpy as np
import os
import torch
import cvzone

from ultralytics import YOLO
from Sort import *

def checkDevice():
    # Test cuda availability
    try:
        torch.cuda.is_available()
    except:
        device = 'cpu'
    else:
        device = 'cuda:0'
    finally:
        print('Running on %s' % device)
        return device
    
def checkVideo(videoPath):
    if not os.path.exists(videoPath):
        print('Video not found')
        exit()
    else:
        video = cv2.VideoCapture(videoPath)
        return video


def detect_car(img, className, pred, tracker):
    detections = np.empty((0, 5))
    for result in pred:
        for box in result.boxes:
            # Get the coordinates of the box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert to int

            # Get the confidence score
            conf = math.ceil(box.conf[0] * 100) / 100

            # Get the predicted class label
            cls = className[int(box.cls[0])]

            if (cls == 'car' or cls == 'truck' or cls == 'bus') and conf > 0.3:
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

    resultsTracker = tracker.update(detections)
    return img, resultsTracker

def counting_cars(track_result, img, limit_line, counter):
    for out in track_result:
        x1, y1, x2, y2, id = out
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        #Draw the label
        cv2.putText(img, f'{int(id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        
        #Count the number of car passes through the limit line
        if limit_line[3] - 25 <= y2 <= limit_line[3] + 25:
            if limit_line[0] <= (x1 + x2) / 2 <= limit_line[2]:
                if id not in counter:
                    counter.append(id)

    return counter
    

def main(videoPath, modelName, graphicPath):
    device = checkDevice()  # Check device for running the model
    model = YOLO(modelName).to(device)  # Load model
    video = checkVideo(videoPath)  # Load video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    #Save video
    out = cv2.VideoWriter('result.mp4', fourcc, 30, (1280, 720))
    graphic = cv2.imread(graphicPath, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread('mask.png')
    limit_line = [250, 380, 750, 380]

    classes = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]  # class list for COCO dataset

    #Tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    counter = []

    # Loop
    while True:
        success, frame = video.read()  # Read frame
        if not success:
            break
        
        #putting the mask
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        frame_mask = cv2.bitwise_and(frame, mask_resized)

        # Detect
        results = model(frame_mask,verbose=False) # result list of detections

        # Draw
        frame, track_result = detect_car(frame, classes, results, tracker)

        # Add line
        cv2.line(frame, (limit_line[0], limit_line[1]), (limit_line[2], limit_line[3]), (0, 0, 255), 4)

        #Counting
        counter = counting_cars(track_result,frame ,limit_line, counter)
        
        #Add counter outcome
        frame = cvzone.overlayPNG(frame, graphic, (0, 0))
        cv2.putText(frame, str(len(counter)), (225, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 7)
        
        # Show
        cv2.imshow('frame', frame)
        out.write(frame)
        cv2.waitKey(1)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close all windows
    cv2.destroyAllWindows()
    os.system('clear')


if __name__ == '__main__': 
    videoPath = 'cars.mp4'
    modelName = 'yolov8n.pt'
    graphicPath = 'graphics.png'
    main(videoPath, modelName, graphicPath)