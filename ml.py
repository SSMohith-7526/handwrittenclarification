pip install opencv-python-headless matplotlib numpy
import cv2
import numpy as np
import matplotlib.pyplot as plt

def clarify_handwritten_image(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Image not found!")
        return
    original = image.copy()

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian Blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Adaptive Thresholding for better contrast
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Step 5: Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 6: Edge enhancement using Canny Edge Detection
    edges = cv2.Canny(cleaned, 50, 150)

    # Step 7: Combine edges with the original image for enhanced clarity
    clarified = cv2.bitwise_or(edges, cleaned)

    # Display results
    images = [original, gray, blurred, thresh, clarified]
    titles = ['Original Image', 'Grayscale', 'Blurred', 'Thresholded', 'Clarified Image']

    plt.figure(figsize=(10, 8))
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray' if i > 0 else None)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Save the clarified image
    output_path = "clarified_image.jpg"
    cv2.imwrite(output_path, clarified)
    print(f"Clarified image saved as {output_path}!")

# Example usage
clarify_handwritten_image("/content/download.jpeg")
