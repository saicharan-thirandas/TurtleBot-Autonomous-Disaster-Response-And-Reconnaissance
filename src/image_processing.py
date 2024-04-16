import cv2
import numpy as np


def draw_bbox(image, corners, center, tag_fam):

    (ptA, ptB, ptC, ptD) = corners

    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))

    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)

    (cX, cY) = (int(center[0]), int(center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
    cv2.putText(
        image, 
        tag_fam, 
        (ptA[0], ptA[1] - 15), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 255, 0), 
        2
    )
    return image