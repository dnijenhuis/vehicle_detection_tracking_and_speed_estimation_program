# This module's purpose is to determine the pixel/line locations of a video. This can in turn be used
# to determine the desired location of the start and end line.
# It needs an image which can be created through the module 'take_one_frame'.

import cv2

IMAGE_PATH = r"PLACEHOLDER"  # Image directory.
image = cv2.imread(IMAGE_PATH)

# Resize the image because the video is 4K, but my personal monitor is, unfortunately, not.
scale_percent = 50  # Percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        original_x = int(x * 100 / scale_percent)
        original_y = int(y * 100 / scale_percent)

        print(f"Pixel Location in Resized Image: (X: {x}, Y: {y})")
        print(f"Mapped Pixel Location in Original Image: (X: {original_x}, Y: {original_y})")
        print(f"Pixel Color (BGR): {image[original_y, original_x]}")
        cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(resized_image, f"({original_x},{original_y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Image", resized_image)


cv2.imshow("Image", resized_image)
cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
