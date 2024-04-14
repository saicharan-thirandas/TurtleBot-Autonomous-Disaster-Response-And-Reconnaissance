import cv2
import numpy as np

def draw_image(occ_map_img, image, current_pose, new_detections):

    image = cv2.circle(image, (int(current_pose[0]), int(current_pose[1])), 3, (255, 0, 0), -1)
    
    for xy, id in new_detections.items():
        image = cv2.circle(image, (int(xy[0]), int(xy[1])), 3, (0, 0, 255), -1)
        image = cv2.putText(image, str(id), (int(xy[0]), int(xy[1])), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2)
    
    return np.clip(cv2.addWeighted(occ_map_img, 1.0, image, 1.0, 0.0), 0, 255)