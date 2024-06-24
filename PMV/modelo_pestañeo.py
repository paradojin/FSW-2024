import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_data(data_path):
    data = []
    labels = []
    for category in ["Closed", "Open"]:
        category_path = os.path.join(data_path, category)
        label = 1 if category == "Closed" else 0

        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (24, 24))
            data.append(image)
            labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    data = np.expand_dims(data, axis=-1)
    labels = np.array(labels)

    return train_test_split(data, labels, test_size=0.2, random_state=42)

data_path = "drowsiness/train"
X_train, X_test, y_train, y_test = load_data(data_path)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
model.save("eye_status_model.h5")