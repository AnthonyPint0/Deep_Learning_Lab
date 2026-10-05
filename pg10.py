import tensorflow as tf
import matplotlib.pyplot as plt


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0


# -------- Without Dropout --------
model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model1.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)


# -------- With Dropout --------
model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model2.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)


# -------- Accuracy Comparison --------
plt.plot(
    history1.history['val_accuracy'],
    label='Without Dropout'
)

plt.plot(
    history2.history['val_accuracy'],
    label='With Dropout'
)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend()

plt.show()


# -------- Loss Comparison --------
plt.plot(
    history1.history['val_loss'],
    label='Without Dropout'
)

plt.plot(
    history2.history['val_loss'],
    label='With Dropout'
)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.legend()

plt.show()