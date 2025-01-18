# **README**

## **Project Overview**
This repository was created for a Computer Vision project conducted as part of my studies. The project focuses on detecting and tracking vehicles in a video, and measuring their speed. Vehicles that are speeding, are flagged with a red box. 

## **Features**
- **Input Video:** The program accepts a `.mp4` video file as input.
- **Vehicle Detection and Tracking:** Utilizes Faster R-CNN and ByteTrack algorithms trained on the COCO dataset to identify and track vehicles.
- **Speed Estimation:** Calculates the speed of each detected vehicle based on user-defined parameters.
- **Output:**
  - Vehicles exceeding the allowed speed limit are flagged with a red bounding box.
  - Non-speeding vehicles are flagged with a green bounding box.
  - The processed video includes visual markers for the distance over which speed is calculated.
  - A `.csv` file stores speed data for all tracked vehicles.

## **Modules**
1. **Main Module:**
   - Contains the core logic of the program;
   - Key functionalities include:
     - Importing necessary external and internal modules;
     - Initializing BYTETrack functionality;
     - Setting up a CSV file to store speed data;
     - Retrieving video input and printing its properties;
     - Configuring the format and parameters for the output video;
     - Running the main processing loop for video analysis.

2; **Parameters Module:**
   - Stores all configurable parameters necessary for video processing;
   - Includes:
     - Paths to relevant files;
     - Speed limits and measuring distances;
     - Pixel coordinates for start and end lines in the video;
     - Pre-processing variables;
     - Configuration for detection and tracking models.
     
3. **Take One Frame Module:**
   - Extracts a single frame from a specified video;
   - The extracted frame can be used by the 'determine_pixel_location' module to define measurement line positions.

4. **Determine Pixel Location Module:**
   - Assists in defining the pixel or line locations in a video. This is necessary to determine where in the video the measurement starts and ends.

## **Limitations**
- **Large Vehicle Detection:** The program currently struggles with recognizing trucks and vans accurately.

## **Future Improvements**
- Enhance the model for improved large vehicle recognition;
- Add support for different video formats (e.g., `.avi`, `.mov`);
- Link the 'parameters' module to the two support modules so that line pixel coordinates can be automatically stored as variables by clicking on locations in the frame. Now, the pixel coordinates still have to be manually added to the 'parameters' module. 