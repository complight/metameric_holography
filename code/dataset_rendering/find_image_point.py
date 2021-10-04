import cv2
import matplotlib.pyplot as plt
import sys

# This simple script loads an image, displays it and then allows you to select a point.
# It then prints out the location of the point in pixel coordinates and normalised coordinates.

image = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)

fig = plt.figure()
plt.imshow(image)
pixel_point = fig.ginput(1)[0]

normalized_point = (pixel_point[0] / image.shape[1], pixel_point[1] / image.shape[0])

print("Pixel coords", pixel_point)
print("Normalized coords", normalized_point)
