pip install tensorflow

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


train_images = train_images / 255.0
test_images = test_images / 255.0


model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),      
    layers.Dense(128, activation='relu'),     
    layers.Dense(10, activation='softmax')     
])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



model.fit(train_images, train_labels, epochs=5)


test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")




plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.show()

prediction = model.predict(test_images[:1])
print("Predicted digit:", prediction.argmax())