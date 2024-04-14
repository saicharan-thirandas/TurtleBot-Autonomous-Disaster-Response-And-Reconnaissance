import cv2
import numpy as np
import matplotlib.pyplot as plt


def filter_black(image):
    # Threshold the grayscale image to find black regions
    _, thresholded = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    print(str(image))
    
    # Invert the thresholded image
    inverted = cv2.bitwise_not(thresholded)
    
    return inverted


def find_closest_pixel_alldist(current_position, image):
    # Find all non-black pixels (pixel value > 0) in the image
    non_black_pixels = np.argwhere(image > 10)
    
    if len(non_black_pixels) == 0:
        # If there are no non-black pixels, return None
        return None
    
    # Calculate distances from the current position to all non-black pixels
    distances = np.linalg.norm(non_black_pixels - current_position, axis=1)
    
    # Find the index of the closest non-black pixel
    closest_index = np.argmin(distances)
    
    # Get the coordinates of the closest non-black pixel
    closest_pixel = non_black_pixels[closest_index]
    
    return closest_pixel

def find_closest_pixel_knn(current_position, image):
    # Find all non-black pixels (pixel value > 10) in the image
    non_black_pixels = np.argwhere(image > 10)
    
    if len(non_black_pixels) == 0:
        # If there are no non-black pixels, return None
        return None
    
    # Reshape non_black_pixels to fit the k-means input format
    non_black_pixels = np.float32(non_black_pixels)
    
    # Perform k-means clustering with 4 clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1)
    _, labels, centers = cv2.kmeans(non_black_pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Find the closest center to the current position
    closest_center_index = np.argmin(np.linalg.norm(centers - current_position, axis=1))
    closest_center = centers[closest_center_index]
    
    return closest_center.astype(int)  # Return as integer coordinates



import numpy as np
import cv2

def find_closest_pixel(current_position, image):
    # Find all non-black pixels (pixel value > 10) in the image
    non_black_pixels_indices = np.argwhere(image > 10)
    
    if len(non_black_pixels_indices) == 0:
        # If there are no non-black pixels, return None
        return None
    
    print("total pixels" +str(non_black_pixels_indices))
    # Randomly sample 10% of non-black pixels
    num_samples = max(1, len(non_black_pixels_indices) // 10)  # Ensure at least one sample
    sampled_indices = np.random.choice(len(non_black_pixels_indices), num_samples, replace=False)
    sampled_pixels = non_black_pixels_indices[sampled_indices]
    
    # Reshape sampled pixels to fit the k-means input format
    sampled_pixels = np.float32(sampled_pixels)
    
    # Perform pairwise distance calculation
    distances = np.linalg.norm(sampled_pixels - current_position, axis=1)
    
    # Find the index of the minimum distance
    min_distance_index = np.argmin(distances)
    
    # Get the closest pixel coordinates
    closest_pixel = sampled_pixels[min_distance_index]
    
    return closest_pixel.astype(int)  # Return as integer coordinates




def main(current_position, image1, image2):
    # Filter out black values in both images
    filtered_image1 = filter_black(image1)
    filtered_image2 = filter_black(image2)
    print("value "+str(filtered_image1))
    
    # Calculate the absolute difference between filtered images
    diff_image = cv2.absdiff(filtered_image1, filtered_image2)
    diff_image = cv2.bitwise_xor(filtered_image1, filtered_image2)
    #print

    
    # Find the closest pixel in the difference image
    closest_pixel = find_closest_pixel(current_position, diff_image)
    
    if closest_pixel is not None:
        # Set the closest non-black pixel as the new target position
        new_target_position = closest_pixel
    else:
        # If no non-black pixels found, set new_target_position to None
        new_target_position = None
    
    return new_target_position, filtered_image1, filtered_image2 ,diff_image

if __name__ == "__main__":
    # Example usage
    current_position = np.array([399, 320])  # Example current position
    image1 = cv2.imread("lidar_map.png", cv2.IMREAD_GRAYSCALE)  # Load image 1 as grayscale
    image2 = cv2.imread("camera_map.png", cv2.IMREAD_GRAYSCALE)  # Load image 2 as grayscale
    
    # Find the new target position and filtered images
    new_target_position, filtered_image1, filtered_image2, diff_image = main(current_position, image1, image2)
    
    # Draw circles on image1 to indicate current and final positions
    image1_with_positions = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    cv2.circle(image1_with_positions, (current_position[1], current_position[0]), 5, (0, 0, 255), -1)  # Red circle for current position
    if new_target_position is not None:
        cv2.circle(image1_with_positions, (new_target_position[1], new_target_position[0]), 5, (0, 255, 0), -1)  # Green circle for final position
    
    # Display both filtered images side by side
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(filtered_image1, cmap='gray')
    axes[0].set_title("Current Lidar Map")
    axes[0].axis('off')
    axes[1].imshow(filtered_image2, cmap='gray')
    axes[1].set_title("Frontier seen by Camera")
    axes[1].axis('off')
    axes[2].imshow(diff_image, cmap='gray')
    axes[2].set_title("Frontier to explore")
    axes[2].axis('off')
    axes[3].imshow(cv2.cvtColor(image1_with_positions, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Current - Red vs Goal - Green ")
    axes[3].axis('off')
    plt.show()

    # Keep the plot open until Ctrl + Z is pressed
    print("Press Ctrl + Z to close the plot.")
    try:
        while True:
            plt.pause(0.1)  # Pause to allow checking for keyboard input
    except KeyboardInterrupt:
        print("Plot closed.")
