import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize data
X_train = X_train / 255.0 
X_test = X_test / 255.0

# Add channel dimension
X_train = X_train[..., np.newaxis] 
X_test = X_test[..., np.newaxis]

# CNN model
model = tf.keras.Sequential([ tf.keras.layers.Conv2D(32, 3, activation='relu',
input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(10, activation='softmax')
])
 
# Compile model 
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=3)


# Test accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Prediction
prediction = model.predict(X_test) 
prediction = np.argmax(prediction, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, prediction)


print("Confusion Matrix:") 
print(cm)

# Display 
ConfusionMatrixDisplay(cm).plot() 
plt.show()
