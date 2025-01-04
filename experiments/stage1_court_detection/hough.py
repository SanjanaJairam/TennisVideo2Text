import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess the image by converting it to grayscale and applying Gaussian Blur.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        tuple: Original image and preprocessed grayscale blurred image.
    """
    img = cv2.imread(image_path)  # Load the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    return img, blur

def detect_edges(blurred_image, low_threshold=50, high_threshold=150):
    """
    Detect edges in the blurred image using Canny edge detection.

    Parameters:
        blurred_image (ndarray): Blurred grayscale image.
        low_threshold (int): Lower threshold for Canny edge detection.
        high_threshold (int): Upper threshold for Canny edge detection.

    Returns:
        ndarray: Edge-detected image.
    """
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold, apertureSize=3)
    return edges

def detect_and_draw_lines(original_image, edges, threshold=160):
    """
    Detect lines using the Hough Transform and draw them on the original image.

    Parameters:
        original_image (ndarray): Original input image.
        edges (ndarray): Edge-detected image.
        threshold (int): Threshold for the Hough line detection.

    Returns:
        ndarray: Image with detected lines drawn.
    """
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    if lines is not None:
        for rho, theta in lines[:, 0]:  # Iterate through each line
            # Calculate line endpoints
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Draw the line on the original image
            cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return original_image

def main():
    """
    Main function to execute the image processing pipeline.
    """
    image_path = '../../data_input/tennis1.png'

    # Preprocess the image
    img, blurred = preprocess_image(image_path)

    # Detect edges
    edges = detect_edges(blurred)

    # Uncomment to visualize intermediate steps if needed
    # cv2.imshow('Blurred Image', blurred)
    # cv2.imshow('Edges', edges)
    # cv2.waitKey(0)

    # Detect and draw lines
    output_img = detect_and_draw_lines(img, edges)

    # Display the final result
    cv2.imshow("Output", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
