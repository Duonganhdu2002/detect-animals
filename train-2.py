import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from imageai.Detection import ObjectDetection


data_dir = './animal-2/'

decoder = ['buffalo', 'elephant', 'rhino', 'zebra']

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
detector.loadModel()

def prepare_data_with_detection(animal, label):
    df, label_lst = [], []
    animal_path = os.path.join(data_dir, animal)
    if not os.path.exists(animal_path):
        print(f"Directory {animal_path} does not exist.")
        return pd.DataFrame(), pd.DataFrame({'label': []})
    
    for img_name in os.listdir(animal_path):
        if img_name.endswith('.jpg'):
            try:
                print(f"Reading file: {img_name}")
                img_path = os.path.join(animal_path, img_name)
                detections = detector.detectObjectsFromImage(input_image=img_path, output_image_path=os.path.join(animal_path, "detected_" + img_name))
                
                detected_animals = [obj["name"] for obj in detections]
                if any(animal in detected_animals for animal in decoder):
                    img = mpimg.imread(img_path)
                    img_resized = resize(img, (128, 128), anti_aliasing=True)
                    df.append(img_resized.reshape(49152))
                    label_lst.append(label)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    
    if not df:
        print(f"No images found for {animal}.")
    
    return pd.DataFrame(df), pd.DataFrame({'label': label_lst})

df_buffalo, label_buffalo = prepare_data_with_detection('buffalo', 0)
df_elephant, label_elephant = prepare_data_with_detection('elephant', 1)
df_rhino, label_rhino = prepare_data_with_detection('rhino', 2)
df_zebra, label_zebra = prepare_data_with_detection('zebra', 3)

X = pd.concat([df_buffalo, df_elephant, df_rhino, df_zebra], axis=0)
y = pd.concat([label_buffalo, label_elephant, label_rhino, label_zebra], axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = X_train.to_numpy().reshape(-1, 128, 128, 3)
X_test = X_test.to_numpy().reshape(-1, 128, 128, 3)
y_train = y_train.to_numpy().astype('int64').reshape(-1, 1)
y_test = y_test.to_numpy().astype('int64').reshape(-1, 1)

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=X_train[0].shape),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

print("Starting training process...")
hist = model.fit(X_train, y_train, batch_size=20, epochs=6, verbose=1, validation_data=(X_test, y_test))
print("Training completed.")

model.save('animal_classifier_model.h5')
print("Model saved as 'animal_classifier_model.h5'.")

plt.title('Model Accuracy')
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='validation')
plt.legend()
plt.show()

plt.title('Model Loss')
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.legend()
plt.show()

y_pred_train = (model.predict(X_train) > 0.5).astype('int')
y_pred_test = (model.predict(X_test) > 0.5).astype('int')

y_train_labels = [[decoder[i] for i, val in enumerate(label) if val] for label in y_train]
y_test_labels = [[decoder[i] for i, val in enumerate(label) if val] for label in y_test]
y_pred_train_labels = [[decoder[i] for i, val in enumerate(label) if val] for label in y_pred_train]
y_pred_test_labels = [[decoder[i] for i, val in enumerate(label) if val] for label in y_pred_test]

train_mat = confusion_matrix(y_train_labels, y_pred_train_labels)
test_mat = confusion_matrix(y_test_labels, y_pred_test_labels)

plot_confusion_matrix(train_mat, figsize=(5, 5), colorbar=True)
plot_confusion_matrix(test_mat, figsize=(5, 5), colorbar=True)

def test_single_image(image_path):
    try:
        print(f"Testing with image: {image_path}")
        img = mpimg.imread(image_path)
        img_resized = resize(img, (128, 128), anti_aliasing=True)
        img_reshaped = img_resized.reshape(1, 128, 128, 3)
        prediction = (model.predict(img_reshaped) > 0.5).astype('int')
        predicted_labels = [decoder[i] for i, val in enumerate(prediction[0]) if val]
        print(f"Predicted labels: {predicted_labels}")
    except Exception as e:
        print(f"Error testing image {image_path}: {e}")

test_single_image('./zebra+and+elephant.png')
