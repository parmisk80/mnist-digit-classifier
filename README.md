# 🔢 Handwritten Digit Recognition with TensorFlow  

This project demonstrates how to build and train a simple neural network using TensorFlow/Keras to recognize handwritten digits from the MNIST dataset.  

---

## 📂 Dataset  

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels:  
- 60,000 images for training  
- 10,000 images for testing  

---

## ⚙️ Model Architecture  

The neural network is built using the Sequential API:  

1. Flatten Layer – converts each 28x28 image into a 1D array of size 784.  
2. Dense Layer (128 neurons, ReLU) – hidden layer that learns patterns in the data.  
3. Dense Layer (10 neurons, Softmax) – output layer that predicts probabilities for digits 0–9.  

---

## 🧠 Training  

- Optimizer: Adam  
- Loss Function: Sparse Categorical Crossentropy  
- Metric: Accuracy  
- Epochs: 5  

```python
model.fit(train_images, train_labels, epochs=5)
