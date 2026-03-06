import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Import Stage 1 data loading and preprocessing
# Skip Stage 2 feature extraction
from data_pipeline import load_and_preprocess

# Define CNN Architecture (3 Convolutional Layers)
class TrafficSignCNN(nn.Module):
    def __init__(self):
        super(TrafficSignCNN, self).__init__()

        # Layer 1: Convolution -> ReLU (Rectified Linear Unit) -> MaxPool
        # Input: 3 channels (RGB), Output: 16 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves image size (64->32)
        
        # Layer 2: Convolution -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves image size (32->16)
        
        # Layer 3: Convolution -> ReLU -> MaxPool (Final allowed conv layer!)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves image size (16->8)
        
        # Fully Connected (Dense) Layers for Classification
        # 64 feature maps * 8 height * 8 width = 4096 flat numbers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4) # Output: 4 classes (Traffic Sign Categories)

    def forward(self, x):
        # Pass data through the network
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x) # Raw scores (logits) for the 4 classes
        return x
    
def prepare_pytorch_data(X_train, X_test, y_train, y_test, batch_size=32):
    print("Converting NumPy arrays to PyTorch Tensors...")

    # Map values from 0-255 to 0-1 (Deep Learning convention)
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Convert from OpenCV formattting to PyTorch formatting
    # (Height, Width, Channels)  to (Channels, Height, Width)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # Tensor Conversion
    tensor_X_train = torch.tensor(X_train)
    tensor_y_train = torch.tensor(y_train, dtype=torch.long)
    tensor_X_test = torch.tensor(X_test)
    tensor_y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create dataloaders (feeds data to model in small batches to save RAM)
    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Main Script
if __name__ == "__main__":
    # Load raw data
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess()

    # Prep for PyTorch
    train_loader, test_loader = prepare_pytorch_data(X_train, X_test, y_train, y_test)

    # Initialise model, loss function, and optimiser
    model = TrafficSignCNN()
    criterion = nn.CrossEntropyLoss() # Standard for multi-class classification
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    # Set constraints
    epochs = 20
    loss_history = []

    print(f"\n Starting CNN Training for {epochs} Epochs")

    # Training Loop
    for epoch in range(epochs):
        model.train()   # Set model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            optimiser.zero_grad()   # Clear old gradients
            outputs = model(images) # Forward pass through network (predict)
            loss = criterion(outputs, labels)   # Calculate error
            loss.backward()         # Backward pass tghrough network (calculate gradients)
            optimiser.step()        # Update weights

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    print("Training Complete!")

    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), loss_history, marker='o', color='red')
    plt.title("CNN Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Cross Entropy)")
    plt.grid(True)
    plt.show()

    # Evaluation using Test Data
    print("\nEvaluating CNN...")
    model.eval()    # Set model to evaluation mode
    all_preds = []
    all_targets = []

    with torch.no_grad():   # Disable gradient tracking to save RAM
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)   # Get highest score
            all_preds.extend(predicted.numpy())
            all_targets.extend(labels.numpy())

        # Calcualte Metrics
        acc = accuracy_score(all_targets, all_preds)
        print(f"CNN Overall Accuracy: {acc * 100:.2f}%\n")  # Accuracy as percentage to 2 dec place
        print("Classification Report: ")
        print(classification_report(all_targets, all_preds, target_names=class_names))

        # Plot Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Reds, ax=ax, xticks_rotation=45)  # Use red to distinguise from Classical
        plt.title("Confusion Matrix: CNN")
        plt.tight_layout()
        plt.show()