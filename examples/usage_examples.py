"""
Example Usage of Image Processing Library
==========================================

This file demonstrates how to use the core image processing functions
programmatically without the GUI.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.core import image_processing as ip

def example_basic_operations():
    """Demonstrate basic image operations"""
    # Load an image
    img = cv2.imread('images/raw/sample.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not load image. Please check the path.")
        return
    
    # Apply histogram equalization
    img_eq = ip.ecualizacion(img)
    
    # Apply gamma correction
    img_gamma = ip.funcion_y_preview(img, gamma=2.2, grises=True, estandarizar=True)
    
    # Apply thresholding
    img_thresh = ip.funcion_umbral_preview(img, umbral=128, grises=True, estandarizar=True)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(img_eq, cmap='gray')
    axes[0, 1].set_title('Equalized')
    axes[1, 0].imshow(img_gamma, cmap='gray')
    axes[1, 0].set_title('Gamma Corrected')
    axes[1, 1].imshow(img_thresh, cmap='gray')
    axes[1, 1].set_title('Thresholded')
    
    plt.tight_layout()
    plt.show()

def example_filtering():
    """Demonstrate filtering operations"""
    img = cv2.imread('images/raw/sample.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not load image.")
        return
    
    # Create a Gaussian kernel
    kernel_size = 5
    sigma = 1.0
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Apply Gaussian filter
    img_filtered = ip.mascara(img, kernel, "Gaussiano", grises=True, estandarizar=True)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(img_filtered, cmap='gray')
    axes[1].set_title('Gaussian Filtered')
    
    plt.tight_layout()
    plt.show()

def example_edge_detection():
    """Demonstrate edge detection"""
    img = cv2.imread('images/raw/sample.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not load image.")
        return
    
    # Sobel edge detection
    sobel_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    
    edges_h = ip.mascara(img, sobel_h, "Sobel Horizontal", grises=True, estandarizar=True, prewitt=True)
    edges_v = ip.mascara(img, sobel_v, "Sobel Vertical", grises=True, estandarizar=True, prewitt=True)
    
    # Combine edges
    edges = np.sqrt(edges_h**2 + edges_v**2)
    edges = ((edges - edges.min()) / (edges.max() - edges.min()) * 255).astype(np.uint8)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(edges_h, cmap='gray')
    axes[1].set_title('Horizontal Edges')
    axes[2].imshow(edges_v, cmap='gray')
    axes[2].set_title('Vertical Edges')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Image Processing Examples")
    print("=" * 50)
    print("\n1. Basic Operations")
    print("2. Filtering")
    print("3. Edge Detection")
    
    choice = input("\nSelect an example (1-3): ")
    
    if choice == "1":
        example_basic_operations()
    elif choice == "2":
        example_filtering()
    elif choice == "3":
        example_edge_detection()
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
