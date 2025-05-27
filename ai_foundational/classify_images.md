Here’s a detailed step-by-step guide for **Project 2: Classify Images (Computer Vision)** using both **TensorFlow/Keras** and **PyTorch**:

---

### **Step 1: Set Up Your Environment**
1. **Install Libraries**:
   ```bash
   # TensorFlow/Keras
   pip install tensorflow matplotlib sklearn

   # PyTorch (visit https://pytorch.org/ for OS-specific installation)
   pip install torch torchvision matplotlib sklearn
   ```

---

### **Step 2: Load and Preprocess the MNIST Dataset**
#### **TensorFlow/Keras**:
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add channel dimension (for TensorFlow Conv2D layers)
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

print("Train shape:", X_train.shape)  # (60000, 28, 28, 1)
print("Test shape :", X_test.shape)   # (10000, 28, 28, 1)
```

#### **PyTorch**:
```python
import torch
from torchvision import datasets, transforms

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0, 1] and adds channel dimension
])

# Load dataset
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# Create data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

---

### **Step 3: Build the Neural Network**
#### **TensorFlow/Keras**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28, 28, 1)),  # Convert 28x28 image to 784 pixels
    Dense(128, activation='relu'),     # Hidden layer with 128 neurons
    Dense(10, activation='softmax')    # Output layer (10 classes)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
```

#### **PyTorch**:
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer
        self.fc2 = nn.Linear(128, 10)      # Output layer

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
```

---

### **Step 4: Train the Model**
#### **TensorFlow/Keras**:
```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2,
    batch_size=64
)
```

#### **PyTorch**:
```python
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Print loss every epoch
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

### **Step 5: Evaluate the Model**
#### **TensorFlow/Keras**:
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np

y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
```

#### **PyTorch**:
```python
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Test Accuracy: {correct / total:.4f}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np

all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_targets.extend(target.numpy())

cm = confusion_matrix(all_targets, all_preds)
print("Confusion Matrix:\n", cm)
```

---

### **Step 6: Visualize Results**
#### **Plot Predictions (TensorFlow/PyTorch)**:
```python
import matplotlib.pyplot as plt

# Plot 9 test images with predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
    plt.axis('off')
plt.show()

# Plot confusion matrix
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

---

### **Key Concepts Explained**
1. **Neural Network Layers**:
   - **Flatten**: Converts 2D image (28x28) to 1D vector (784).
   - **Dense/Linear**: Fully connected layer with weights and biases.
   - **Activation Functions**:
     - `ReLU`: Introduces non-linearity (outputs `max(0, x)`).
     - `Softmax`: Converts logits to probabilities (sums to 1).

2. **Training Loop**:
   - **Forward Pass**: Compute predictions (`model(inputs)`).
   - **Loss Calculation**: Compare predictions with true labels (`cross-entropy`).
   - **Backward Pass**: Compute gradients (`loss.backward()`).
   - **Optimizer Step**: Update weights (`optimizer.step()`).

3. **Loss Function**:
   - **Cross-Entropy**: Measures difference between predicted probabilities and true labels.

4. **Optimizers**:
   - **Adam**: Adaptive learning rate optimizer (combines RMSProp and momentum).
   - **SGD**: Stochastic Gradient Descent (vanilla or with momentum).

---

### **Improving the Model**
1. **Add Convolutional Layers** (better for images):
   ```python
   # TensorFlow
   model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

   # PyTorch
   self.conv1 = nn.Conv2d(1, 32, 3, 1)
   ```
2. **Hyperparameter Tuning**:
   - Adjust learning rate, batch size, epochs.
   - Use `GridSearchCV` (for scikit-learn wrappers) or manual trials.
3. **Regularization**:
   - Add dropout layers (`tf.keras.layers.Dropout(0.2)` or `nn.Dropout(0.2)`).

---

### **Resources**
- **TensorFlow/Keras Guide**: [Image Classification](https://www.tensorflow.org/tutorials/keras/classification)
- **PyTorch Tutorial**: [MNIST](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- **MNIST Dataset**: [Yann LeCun’s Website](http://yann.lecun.com/exdb/mnist/)

By following these steps, you’ll build a foundational understanding of neural networks for computer vision! 🖼️🧠
