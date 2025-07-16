# ----------------------------------------------------------------------------#

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------#

def preProcessing(myImage):
    # Converting to grayscale and applying Otsu's thresholding.
    grayImg = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(grayImg, 0, 255, 
                                cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # Noise removal using morphological opening.
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    
    # Character segmentation.
    result_img = character_seg_rajput(cleaned, myImage.copy())
    
    return result_img

# ----------------------------------------------------------------------------#

def character_seg_rajput(binary_img, output_img):
    # Calculating the vertical projection.
    v_projection = np.sum(binary_img // 255, axis=0)

    # Smoothing the projection profile with a large kernel.
    kernel_size = 7
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(v_projection, kernel, 'same')

    # Dual-threshold approach for a more robust gap detection.
    low_thresh = 0.05 * np.max(smoothed)
    high_thresh = 0.18 * np.max(smoothed)
    gap_mask = (smoothed < low_thresh) | (smoothed < high_thresh)

    # Find gap regions.
    gap_starts, gap_ends = find_gap_regions(gap_mask, binary_img.shape[1])

    # Determining character regions.
    char_regions = []
    start = 0
    height, width = binary_img.shape

    for i in range(len(gap_starts)):
        char_regions.append((start, gap_starts[i]))
        start = gap_ends[i]
    char_regions.append((start, width))  # Final character region.

    # Drawing character bounding boxes.
    for x_start, x_end in char_regions:
        if x_end - x_start < 3:  # Skipping tiny regions.
            continue

        segment = binary_img[:, x_start:x_end]
        row_sum = np.sum(segment, axis=1)
        non_zero_rows = np.where(row_sum > 0)[0]

        if non_zero_rows.size > 0:
            y_top = non_zero_rows.min()
            y_bottom = non_zero_rows.max()
            cv2.rectangle(output_img, 
                          (x_start, y_top), 
                          (x_end, y_bottom), 
                          (0, 255, 0), 2)

    return output_img

# ----------------------------------------------------------------------------#

def find_gap_regions(gap_mask, image_width):
    """
    Identifies contiguous gap regions
    with adaptive width filtering
    """
    gap_changes = np.diff(gap_mask.astype(int))
    gap_starts = np.where(gap_changes == 1)[0] + 1
    gap_ends = np.where(gap_changes == -1)[0] + 1

    # Handling edge cases.
    if gap_mask[0]:
        gap_starts = np.insert(gap_starts, 0, 0)
    if gap_mask[-1]:
        gap_ends = np.append(gap_ends, len(gap_mask) - 1)

    # Adaptive minimum gap width.
    min_gap_width = max(2, image_width // 200)

    valid_gaps = []
    for i in range(len(gap_starts)):
        if gap_ends[i] - gap_starts[i] >= min_gap_width:
            valid_gaps.append((gap_starts[i], gap_ends[i]))

    if valid_gaps:
        gap_starts, gap_ends = zip(*valid_gaps)
    else:
        gap_starts, gap_ends = [], []

    return gap_starts, gap_ends

# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    png = "/home/ml3/Desktop/Thesis/Screenshot_1.png"
    image = cv2.imread(png)
    
    if image is None:
        print("Error: Image not loaded.")
    else:
        result = preProcessing(image)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig("character_segmentation_result.png", 
                   bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()
