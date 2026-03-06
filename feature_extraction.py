import cv2
import numpy as np
from skimage.feature import hog
from data_pipeline import load_and_preprocess  # First stage script

def extract_colour_histogram(image, bins=(8, 8, 8)):
    # Extracts colour histogram from HSV (Hue, Saturation, Value) space and flattens it

    # Convert RGB to HSV (better for real-world conditions)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Calculate histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # Normalise histogram (stops differing lighting intensity from breaking the model)
    cv2.normalize(hist, hist)
    # Flatten into a 1 Dimension array
    return hist.flatten()

def extract_hog_features(image):
    # Extracts the histogram of oriented gradients (HOG) to capture shape and edges
    # Combines calculation of gradients and orientation and sorting continuous data 
    # into discrete bins using a composite algorithm.

    # HOG needs a greyscale image
    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate HOG features
    features = hog(grey_image, 
                   orientations=9, 
                   pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), 
                   transform_sqrt=True, 
                   block_norm="L2")
    return features

def extract_features(X):
    # Loops through all images and combines the Colour and HOG features into one vector.

    print(f"Extracting features from {len(X)} images...")
    features=[]

    for img in X:
        colour_feat = extract_colour_histogram(img)
        hog_feat = extract_hog_features(img)

        # Combine both features into a single 1 Dimension array per image
        combined_features = np.hstack([colour_feat, hog_feat])
        features.append(combined_features)

    return np.array(features)

if __name__ == "__main__":
    # Load data from Stage 1 script
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess()

    # Extract features
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)

    print("\n Feature Extraction Complete!")
    print(f"Original image shape: {X_train[0].shape}")
    print(f"New feature vector size per image: {X_train_features.shape[1]}")