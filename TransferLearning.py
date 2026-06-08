import tensorflow as tf
import os
from keras import Sequential
from keras import models, layers
from keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import GlobalAveragePooling2D

objectClasses = 10
imagesHeight = 224
imagesWidth = 224

#Daten aus dem Verzeichnis laden
trainingData = image_dataset_from_directory("Data/train", image_size=(imagesHeight,imagesWidth),batch_size=32,shuffle=True,seed=33)
validationData = image_dataset_from_directory("Data/validation",image_size=(imagesHeight,imagesWidth),batch_size=32)
testData = image_dataset_from_directory("Data/test",image_size=(imagesHeight,imagesWidth),batch_size=32,shuffle=False)
#Klassennamen ausgeben
class_names = trainingData.class_names
print(class_names)

#Daten normalisieren
trainingData = trainingData.map(lambda x, y:(x / 255.0, y))
validationData = validationData.map(lambda x, y:(x / 255.0, y))
testData = testData.map(lambda x, y:(x / 255.0, y))

#Batches sollen gepuffert werden, um Wartezeiten beim Training zu verringern
trainingData = trainingData.prefetch(tf.data.AUTOTUNE)
validationData = validationData.prefetch(tf.data.AUTOTUNE)
testData = testData.prefetch(tf.data.AUTOTUNE)

#Callbacks festlegen:
#Dynamische Reduktion der Lernrate während des Trainings
lrReducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3)
#Training beenden, wenn der Loss sich nicht verbessert
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
#Logging mit TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs/runTL_testInceptionY"
)

#Daten zufaellig veraendern um Generalisierung zu verbessern
dataAugmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

#Model laden
#model = tf.keras.models.load_model("modelTL_testInceptionQ.keras")

#Identifizieren der Bilder bei Klassenverwechselung
# class_path = "C:/Users/mikab/PycharmProjects/DeepLearning/Data/test/Hoopoe(Wiedehopf)"
# true_class = os.path.basename(class_path)
# for filename in os.listdir(class_path):
#     image_path = os.path.join(class_path, filename)
#     img = tf.keras.utils.load_img(
#         image_path,
#         target_size=(224, 224)
#     )
#     img_array = tf.keras.utils.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0
#
#     pred = model.predict(img_array, verbose=0)
#
#     pred_idx = np.argmax(pred[0])
#     pred_class = class_names[pred_idx]
#
#     if pred_class != true_class:
#         confidence = pred[0][pred_idx]
#
#         print("=" * 100)
#         print(f"Bildpfad:   {image_path}")
#         print(f"Echte Klasse: {true_class}")
#         print(f"Vorhersage:   {pred_class}")
#         print(f"Wahrscheinlichkeit:    {confidence:.4f}")



# Ausgabe alle Layers
# for i, layer in enumerate(inceptionV3.layers):
#     print(i, layer.name, layer.trainable)

#Modelle laden
#vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet',input_shape=(imagesHeight,imagesWidth,3),pooling='none')
#resnet34 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',input_shape=(imagesHeight,imagesWidth,3),pooling='none')

inceptionV3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',input_shape=(imagesHeight,imagesWidth,3),pooling='none')

#FineTuning: Freigeben der letzten Schichten
# for layer in inceptionV3.layers[:-30]:
#     layer.trainable = False
# for layer in inceptionV3.layers[-30:]:
#      layer.trainable = True

#Model zusammenbauen
model = Sequential()
model.add(dataAugmentation)
model.add(inceptionV3)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(objectClasses, activation='softmax'))

#Optimierungsfunktion festlegen und Lernrate festlegen
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
#Model kompilieren , Verlustfunktion festlegen.
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'],optimizer=optimizer)
#Training starten, Epochen festlegen
history = model.fit(trainingData,epochs=30,validation_data=validationData, callbacks=[tensorboard_callback,lrReducer,earlyStopping])
#Model speichern
model.save("modelTL_testInceptionY.keras")

#Model auf unbekannte Daten evaluieren
test_loss, test_acc = model.evaluate(testData,verbose=2)
print('Test accuracy:', test_acc)



#Vorhersagen mit Testdaten treffen, Confusion Matrix erzeugen:
# y_pred_probs = model.predict(testData)
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.concatenate([y for x, y in testData], axis=0)
# cm = confusion_matrix(y_true, y_pred)
#
# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=class_names
# )
#
# fig, ax = plt.subplots(figsize=(10, 10))
# disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()

#print(cm)