# This module, the parameters module, contains all the parameters that can (and partly must) be configured when a video
# is to be processed.
#
# After importing the necessary modules, the module defines parameters regarding:
# - Paths of relevant files;
# - Speed limits and measuring distances;
# - Locations in the video frames (pixel values) where the measuring starts and ends;
# - The (calculated) minimum number of frames between the start and end to not be speeding;
# - Pre-processing variables;
# - Variables for the detection software;
# - Variables for the tracking software;
# The module ends with logic regarding the drawing of lines and their labels in the output video. These lines do not
# have any technical function in the program but visually demonstrate to the user where the start and end lines are.

import cv2
from types import SimpleNamespace

# 🔧 Location of the input video, output video and CSV-file. These have to be set manually.
input_video_path = r"PLACEHOLDER"
output_video_path = r"PLACEHOLDER"
csv_path = r"PLACEHOLDER"

# Variables relevant for speed(ing) calculation.
# speed_limit and distance_in_meters have to be input manually.
speed_limit = 100      # 🔧 Set manually for every new setup. Legally allowed speed limit in kilometer per hour.
distance_in_meters = 60  # 🔧 Set manually for every new setup. Distance between start and end lines (in meters).

# 🔧 Define pixels (on the Y-axis) where the lines are drawn. Additionally, the tolerances (buffer lines) are set which
# determine the size of the buffer zone in which starting and ending vehicles are to be detected.
start_line = 501  # 🔧 Set manually for every new setup.
end_line = 949  # 🔧 Set manually for every new setup.
start_line_tolerance = 100  # 🔧 Set manually for every new setup.
end_line_tolerance = 700   # 🔧 Set manually for every new setup.

# FPS is (dynamically) taken from the input video to calculate the minimum_allowed_number_of_frames. So the number of
# frames that it has to take at least to cross the measured distance between start line and end line.
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Now for the logic: (distance_in_meters / speed_limit_in_meters_per_second) results in the minimum number of seconds
# that the vehicle should need to get from start to end. This is then multiplied by FPS to convert from seconds to FPS.
# I originally considered rounding the minimum_allowed_number_of_frames_for_distance down but this is not necessary
# since the number of measured frames will always be an integer, so the driver will either drive more or less than,
# for example, 67.5 frames.
speed_limit_in_meters_per_second = (speed_limit * 1000) / 3600  # Speed limit converted to meters per second.
minimum_allowed_number_of_frames_for_distance = (distance_in_meters / speed_limit_in_meters_per_second) * fps

# Manually checking my logic:
# 80 km per uur = 22.222 meter per second.
# Assuming a distance of 50 meters, this should take at least 2.25 second.
# Assuming 30 FPS this should take at least 67.5 frames.
# I checked this through a print statement using these variables, and the output was as expected.
# print(fps)
# print(minimum_allowed_number_of_frames_for_distance)

# Variables for pre-processing. It is recommended that these variables are not altered since their default values
# are for most cases the most suitable.
GAUSSIAN_KERNEL = (5, 5)  # Size of the kernel. Bij default set to 5 by 5 pixels.
GAUSSIAN_SIGMA = 0      # Standard deviation is automatically calculated when set to 0.
contrast_channel = 0  # Histogram Equalization set to zero results in adding contrast. Setting to 1 or 2 affects color.

# Confidence threshold for detecting vehicles. An increase of this value results in less false positives, but in more
# false negatives. Decrease of this value has the opposite effect.
detection_score_threshold = 0.65

# Vehicle classes from COCO Dataset. Only motorized land vehicles are included.
VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, motor, bus and truck.

# 🔧 Add a max box area threshold: the maximum box size that can be detected. This should be set manually for every new
# video location/installment. Putting a maximum on the box size, prevented boxes (vehicles) being falsely detected that
# were the size of more than 30% of the screen (including multiple vehicles simultaneously). In other words, this
# restriction decreases the chance that 1 box contains more than 1 vehicle, and that 1 vehicle is contained in more
# than 1 box.
max_box_area = 1600000  # 🔧 Set manually for every new setup.

# 🔧 BYTETrack Arguments.
args = SimpleNamespace(
    track_thresh=0.60,          # Confidence threshold. Detections below this value will not be tracked
    track_buffer=fps*5,         # Number of frames that an ID/vehicle is tracked after not being detected, set to 5 sec.
    match_thresh=0.70,          # Threshold for matching across frames. Increase for more flexible matching.
    aspect_ratio_thresh=1.8,    # Defines to what extent stretched boxes are kept or discarded during tracking.
    min_box_area=2500,          # 🔧 Set manually for every new setup. Minimum box size. Has to be set manually based
                                # on the position of the camera and the distance from the camera to the start line.
    mot20=False
)


def add_lines_and_labels(frame, frame_width):
    # Add lines and line labels. This function is called for every frame-processing and adds a start line, an end line,
    # their respective buffer/tolerance lines, and the line labels.
    cv2.line(frame, (0, start_line), (frame_width, start_line), (255, 0, 0), 2)
    cv2.line(frame, (0, end_line), (frame_width, end_line), (255, 0, 0), 2)
    cv2.line(frame, (0, start_line - start_line_tolerance), (frame_width, start_line - start_line_tolerance), (255, 0, 0), 2)
    cv2.line(frame, (0, end_line + end_line_tolerance), (frame_width, end_line + end_line_tolerance), (255, 0, 0), 2)
    cv2.putText(frame, 'Start Line', (50, start_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.putText(frame, 'Start Buffer', (50, (start_line - start_line_tolerance) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.putText(frame, 'End Line', (50, end_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.putText(frame, 'End Buffer', (50, (end_line + end_line_tolerance) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    return frame
