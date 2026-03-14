import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # We need this to convert HSV to RGB for plotting!

# Import your working data loader
from data_pipeline import load_and_preprocess

def plot_average_hue_histograms():
    print("Loading data to calculate average Hue distributions...")
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess()
    
    X_all = np.concatenate((X_train, X_test))
    y_all = np.concatenate((y_train, y_test))
    
    # --- Generate the 180 Colors ---
    # Matplotlib uses a 0.0 to 1.0 scale for HSV, while OpenCV uses 0 to 180.
    # We create an array of 180 colors, mapping the OpenCV scale to Matplotlib's scale.
    hsv_colors_array = np.zeros((180, 3))
    hsv_colors_array[:, 0] = np.arange(180) / 180.0  # Hue (0.0 to 1.0)
    hsv_colors_array[:, 1] = 1.0                     # Saturation (Full)
    hsv_colors_array[:, 2] = 1.0                     # Value (Full)
    
    # Convert those pure HSV values into standard RGB colors so Matplotlib can draw them
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors_array)
    
    # Set up the 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(class_names):
        print(f"Processing {class_name}...")
        
        class_images = X_all[y_all == idx]
        avg_hist = np.zeros((180, 1))
        
        for img in class_images:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            avg_hist += hist
            
        avg_hist /= len(class_images)
        avg_hist_flat = avg_hist.flatten()
        
        # --- The Magic Trick: Colorized Area ---
        # Instead of fill_between, we draw 180 thin vertical bars, each with its true color
        axes[idx].bar(np.arange(180), avg_hist_flat, color=rgb_colors, width=1.0, edgecolor='none', alpha=0.7)
        
        # We still draw the black outline on top so the shape is sharp
        axes[idx].plot(avg_hist_flat, color='black', linewidth=1.5)
        
        axes[idx].set_title(f'Average Hue: {class_name}', fontsize=14, fontweight='bold')
        axes[idx].set_xlim([0, 180])
        axes[idx].set_xlabel('Hue Value (0 - 180)')
        axes[idx].set_ylabel('Average Pixel Count')
        axes[idx].grid(True, linestyle='--', alpha=0.4)

    # --- Add the Master Color Legend ---
    # This creates a color bar on the right side of the entire figure
    cmap = plt.get_cmap('hsv')
    norm = plt.Normalize(vmin=0, vmax=180)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), aspect=30, pad=0.03)
    cbar.set_label('OpenCV Hue Value', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    plt.suptitle("Average Hue Channel Distribution by Traffic Sign Class", fontsize=18, fontweight='bold', y=0.98)
    plt.show()

if __name__ == "__main__":
    plot_average_hue_histograms()