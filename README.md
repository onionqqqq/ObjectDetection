# ObjectDetection

## Introduction
This project aims to detect vehicles and count them using YOLOv8.

* Use a mask to restrict the area so that the model only needs to detect the area on the main road.
* Use Sort to keep track of the vehicles and add an ID to them.
* Add a line to count the number of vehicles passing through it.

## Installation

    git clone https://github.com/onionqqqq/ObjectDetection.git
    cd ObjectDetection
    conda create -n ObjectDetection python=3.8
    conda activate ObjectDetection
    pip install -r requirements.txt

## Reference
[Youtube](https://www.youtube.com/watch?v=WgPbbWmnXJ8)    
[YOLOv8](https://github.com/ultralytics/ultralytics)
