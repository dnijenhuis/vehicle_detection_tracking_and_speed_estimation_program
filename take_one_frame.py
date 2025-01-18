# This module's purpose is to take one frame from a specified video. This frame can then be used by the
# determine_pixel_location module in order to determine the desired location of the start and end line.
# It needs an image which can be created through the module 'take_one_frame'.

import cv2

VIDEO_SOURCE = r"PLACEHOLDER"  # Directory of input video.
OUTPUT_IMAGE = r"PLACEHOLDER"  # Directory of output image.

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise IOError("Cannot open video file")

frame_number = 10  # Adjust to the frame number you need.
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

ret, frame = cap.read()
if ret:
    cv2.imwrite(OUTPUT_IMAGE, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print(f"Frame {frame_number} saved as {OUTPUT_IMAGE}")
else:
    print(f"Failed to read frame {frame_number}")

cap.release()
