import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Import data loader
from data_pipeline import load_and_preprocess

def generate_hog_graphic():
    print("Loading data to generate HOG visualisation...")
    X_train, _, y_train, _, class_names = load_and_preprocess()
    
    # First image in the training set
    sample_image = X_train[120]
    sample_label = class_names[y_train[120]]
    
    # HOG looks at intensity gradients (light vs dark edges), so we convert to greyscale first
    gray_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)
    
    # Calculate HOG and set visualize=True to get the image array back
    features, hog_image = hog(
        gray_image, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=True, 
        feature_vector=True
    )
    
    # Enhance the contrast of the HOG image so it shows up clearly
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    # Set up the side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(sample_image)
    ax1.set_title(f'Original RGB Image\nClass: {sample_label}', fontsize=14, fontweight='bold')
    
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('HOG Feature Visualisation\n(Edge Gradients)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    generate_hog_graphic()