import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_preprocess(data_dir='.\GTSRB_MiniPrj', img_size=64):
    # Load images from folders, resize to uniform size, split into train/test sets

    # Define categories based on folder names (must match)
    categories = ['Positive_Directions', 'Prohibitive_signs', 'Speed_Limit', 'Warnings']

    X = [] # Stores images
    y = [] # Stores classification labels

    print("Loading and preprocessing of images in progress...")

    # For each folder/catergory
    for label_idx, category in enumerate(categories):   # Takes each category and assignes numerical value (label_idx)
                                                        # things like file path require name, but classifier requires a numerical value
        folder_path = os.path.join(data_dir, category)

        if not os.path.exists(folder_path):
            print(f"Could not find folder '{folder_path}'")
            continue
        
        # For each image in the folder
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)  # Create path for each image for CV2 access

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue    # Skip if invalid image file

            # Convert from OpenCV BGR to standard RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize the image for standardisation
            img = cv2.resize(img, (img_size, img_size))

            X.append(img)       # Add image data to "x-axis"
            y.append(label_idx) # Add corresponding label to "y-axis"

    # Convert lists to NumPy arrays (required by some libs)
    X = np.array(X)
    y = np.array(y)

    print(f"Total iamges loaded: {len(X)}")
    print(f"Image shape: {X.shape[1:]}")    # Height, width, colour channels (RGB=3)

    # Train-Test Split (80/20)
    # random_state=42 ensures same split each time we run
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set: {len(X_train)} images")
    print(f"Testing set: {len(X_test)} images")

    return X_train, X_test, y_train, y_test, categories

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess()

    if len(X_train) > 0:
        plt.figure(figsize=(4,4))
        plt.imshow(X_train[0])
        plt.title(f"Label: {class_names[y_train[0]]} ({y_train[0]})")
        plt.axis('off')
        plt.show()




