# This Python program contains two modules: main and parameters. The main module contains the core logic of this
# program, while the parameters module contains all the parameters that can (and have to) be configured when a video
# is to be processed.

# The main module contains the following functionality:
# - Imports the necessary external and internal modules;
# - Starts the BYTETrack functionality;
# - Sets up a CSV-file (now with speed included);
# - Retrieves the video and prints its properties;
# - Determines the format and parameters of the output video;
# - Sets up the needed dictionary and set for vehicle tracking;
# - Executes the while loop containing the core functionality of this program;
# - Releases videos and closes the window.

import cv2  # Necessary for various Computer Vision functionalities (drawing lines, applying filters, etc.)
import torch  # Provides functionality to run Faster R-CNN which is used to detect vehicles.
import torchvision
import csv  # csv, os and datetime libraries are used to create the CSV-file containing speed data.
import os
import datetime
import parameters  # Import the module containing the parameters.
import yolox.tracker.byte_tracker as byte_tracker  # ByteTrack is needed to track the vehicles over time.

# Starting up the BYTETrack functionality and naming the arguments (set in the parameters module).
tracker = byte_tracker.BYTETracker(parameters.args)

# Loading the Faster R-CNN Model with ResNet101 backbone instead of default ResNet50 backbone. This increases
# accuracy. Note: The user could later implement ResNet152 but this requires additional changes in this program.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
    backbone_name="resnet101")
model.eval()
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Making sure that the directory and CSV-file exist and assigning a (unique) name to it based on the current date
# and time. Furthermore, creating the relevant columns for the CSV (including speed).
os.makedirs(parameters.csv_path, exist_ok=True)  # Making the directory.
csv_filename = os.path.join(parameters.csv_path, f"CSV_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.csv")
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Vehicle ID', 'Start Frame', 'End Frame', 'Frames Elapsed', 'Speed'])

# Opens the video and assigns a variable to it. The location of the file has to be set in the 'parameters' module.
cap = cv2.VideoCapture(parameters.input_video_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")  # Error message in case the video cannot be opened.

# Gets video properties (resolution and FPS) and assigns variables to them. Resolution is later used to draw lines,
# determine whether vehicles crossed lines, and to determine the resolution of the output video.
# Also prints the properties for convenience.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Frame_width: {frame_width}")
print(f"Frame_height: {frame_height}")
print(f"FPS: {fps}")

# Functionality to create a new (processed) video file. It is saved to the location set in the parameters file.
# Parameters for the video (resolution and FPS) are the same as the input video. Input and output are in .mp4 format.
# which is a simple format with high compatibility.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(parameters.output_video_path, fourcc, fps, (frame_width, frame_height))

# This code contains the logic for keeping track of the vehicles' crossing times. It is a dictionary where for
# each unique vehicle ID, the frame_ID of the start and the frame_ID of the end of the distance tracking are
# stored, and the color of the box is stored. This dictionary gets 'filled' later in this module.
crossing_times = {}  # {track_id: {'start': frame_id, 'end': frame_id, 'color': (R, G, B)}}
logged_vehicles = set()  # Keeps track of vehicles that have already been logged to prevent duplicates in CSV.

# Frame Processing
frame_id = 0  # Initialize a frame counter to keep track of the number of frames processed.

# The loop below processes the input video frames:
#   - Preprocessing (filter and contrast increase);
#   - Add lines to the output frame;
#   - Convert frames to a format suitable for Faster R-CNN;
#   - Detect vehicles using Faster R-CNN;
#   - Track vehicles using BYTETrack;
#   - Store speed data in the CSV-file.
while cap.isOpened():
    ret, frame = cap.read()  # Read the next frame from the video source.
    if not ret:
        break  # Stop processing if there is no new frame.

    frame_id += 1  # Increment the frame counter for each processed frame.

    # Preprocessing: Gaussian Blur and contrast increase.
    frame = cv2.GaussianBlur(frame, parameters.GAUSSIAN_KERNEL, parameters.GAUSSIAN_SIGMA)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, parameters.contrast_channel] = cv2.equalizeHist(yuv[:, :, parameters.contrast_channel])
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Add lines and line labels as specified in parameters module.
    frame = parameters.add_lines_and_labels(frame, frame_width)

    # Convert the frame to RGB and make it PyTorch-compatible tensor ready for the model.
    img_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)

    # Perform object detection with Faster R-CNN.
    with torch.no_grad():
        predictions = model(img_tensor)

    # Prepare Detections for BYTETrack (filter out low-confidence or non-vehicle objects).
    detections = []
    for i, detection_score in enumerate(predictions[0]['scores']):
        if detection_score > parameters.detection_score_threshold:
            label = predictions[0]['labels'][i].item()
            if label in parameters.VEHICLE_CLASSES:  # Vehicle from the COCO dataset
                bbox = predictions[0]['boxes'][i].numpy()
                x1, y1, x2, y2 = bbox
                detections.append([x1, y1, x2, y2, float(detection_score), label])

    # Filter out bounding boxes that are too small or too large.
    filtered_detections = []
    for det in detections:
        x1, y1, x2, y2, scr, lbl = det
        w = x2 - x1
        h = y2 - y1
        area = w * h
        if area >= parameters.args.min_box_area and area <= parameters.max_box_area:
            filtered_detections.append(det)

    # Convert the filtered detections into a PyTorch tensor (or an empty tensor if no detections).
    if filtered_detections:
        detection_tensor = torch.tensor(filtered_detections, dtype=torch.float32)
    else:
        detection_tensor = torch.empty((0, 6), dtype=torch.float32)

    # Update the tracker with the current frame's detections.
    online_targets = tracker.update(detection_tensor, [frame_height, frame_width], (frame_height, frame_width))

    # Process each tracked vehicle.
    for track in online_targets:
        x1, y1, w, h = map(int, track.tlwh)
        x2, y2 = x1 + w, y1 + h
        bottom_x = x1 + (w // 2)
        bottom_y = y2

        # Get the track's color from crossing_times or assign a default color.
        color = crossing_times.get(track.track_id, {}).get('color', (0, 165, 255))

        # Check if vehicle is in the start zone.
        if (parameters.start_line - parameters.start_line_tolerance) <= bottom_y <= parameters.start_line:
            crossing_times.setdefault(track.track_id, {})['start'] = frame_id  # Last frame
            print(f"In frame no. {frame_id}, vehicle with ID {track.track_id} is in the start zone.")

        # Check if vehicle is in the end zone.
        if parameters.end_line <= bottom_y <= (parameters.end_line + parameters.end_line_tolerance):
            crossing_times.setdefault(track.track_id, {}).setdefault('end', frame_id)  # First frame
            print(f"In frame no. {frame_id}, vehicle with ID {track.track_id} is in the end zone.")

            # If the vehicle has not been logged yet (to avoid duplicates) and has a start frame.
            if track.track_id not in logged_vehicles and 'start' in crossing_times[track.track_id]:
                elapsed_frames = crossing_times[track.track_id]['end'] - crossing_times[track.track_id]['start']

                # Calculate speed based on elapsed frames, distance, and fps
                elapsed_time_sec = elapsed_frames / parameters.fps  # Time in seconds
                if elapsed_time_sec > 0:
                    speed_m_s = parameters.distance_in_meters / elapsed_time_sec   # m/s
                    speed_km_h = speed_m_s * 3.6                                   # km/h
                else:
                    speed_km_h = 0.0  # If, for any reason, frames were 0 (unlikely, but just as a fallback)

                # Assign color based on whether it's speeding or not.
                color = (0, 0, 255) if elapsed_frames < parameters.minimum_allowed_number_of_frames_for_distance \
                    else (0, 255, 0)
                crossing_times[track.track_id]['color'] = color

                # Write the record to CSV, including the speed.
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        track.track_id,
                        crossing_times[track.track_id]['start'],
                        crossing_times[track.track_id]['end'],
                        elapsed_frames,
                        f"{speed_km_h:.1f}"
                    ])
                logged_vehicles.add(track.track_id)

        # Draw rectangle, dot, and ID on the frame.
        cv2.circle(frame, (bottom_x, bottom_y), 6, (0, 255, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {track.track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the real-time processed video in a window.
    cv2.imshow('Vehicle Detection and Tracking', frame)

    # Save the processed frame in the output file.
    out.write(frame)

    # Press 'q' to stop early.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources: input and output videos, and close window.
cap.release()
out.release()
cv2.destroyAllWindows()
