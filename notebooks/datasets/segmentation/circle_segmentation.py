import cv2
import numpy as np

def random_rgb_color():
    return [int(c) for c in np.random.randint(0, 255, size=3)]

def circle_segmentation_sample(width, height, n_polygons=5, n_corners_per_polygon=10):

    img = np.zeros((height, width, 3)).astype(np.uint8)
    mask = np.zeros((height, width, 1)).astype(np.uint8)

    for _ in range(n_polygons):

        color = random_rgb_color()
        pts = np.hstack([np.random.randint(-height, 2*height, size=(n_corners_per_polygon,1)),
                         np.random.randint(-width, 2*width, size=(n_corners_per_polygon,1))])
        pts = np.vstack([pts, pts[-1:]])
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],False,color, thickness=2)

    radius = np.random.randint( int(min(width, height)/8), min(width, height)/4)
    rotation = np.random.randint(0, 360)
    center = np.random.randint(radius, height-radius), np.random.randint(radius, width-radius)
    color = random_rgb_color()

    cv2.circle(img, center, radius=radius, color=color, thickness=-1)
    cv2.circle(mask, center, radius=radius, color=1, thickness=-1)

    return img, mask