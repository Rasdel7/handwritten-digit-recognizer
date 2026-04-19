import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

print(f"TensorFlow version: {tf.__version__}")

# Load MNIST Dataset
print("\nLoading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Training samples : {X_train.shape}")
print(f"Testing samples  : {X_test.shape}")

# Visualize Sample Digits
plt.figure(figsize=(12, 4))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(str(y_train[i]), fontsize=10)
    plt.axis('off')
plt.suptitle('Sample Handwritten Digits from MNIST', fontsize=13)
plt.tight_layout()
plt.savefig('sample_digits.png')
print("Sample digits chart saved!")

# Preprocess Data
X_train = X_train / 255.0
X_test  = X_test  / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

# Build Neural Network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10,  activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the Model
print("\nTraining neural network...")
history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# Training History Chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'],     label='Train', color='#3498db')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='#e74c3c')
axes[0].set_title('Model Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='#3498db')
axes[1].plot(history.history['val_loss'], label='Val',   color='#e74c3c')
axes[1].set_title('Model Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history chart saved!")

# Confusion Matrix
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - Digit Recognition', fontsize=14)
plt.ylabel('Actual Digit')
plt.xlabel('Predicted Digit')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved!")

# Visualize Predictions
plt.figure(figsize=(12, 5))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    pred   = y_pred[i]
    actual = y_test[i]
    color  = 'green' if pred == actual else 'red'
    plt.title(f'P:{pred} A:{actual}', color=color, fontsize=9)
    plt.axis('off')
plt.suptitle('Predictions (Green=Correct, Red=Wrong)', fontsize=13)
plt.tight_layout()
plt.savefig('predictions.png')
print("Predictions chart saved!")

print(f"\nDone! Final accuracy: {test_acc * 100:.2f}%")
print("Check your folder for all 4 charts.")