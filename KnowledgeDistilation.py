import tensorflow as tf
import os
import keras
from keras import models, layers
from keras.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import numpy as np

from Distiller import Distiller
from architectures import KD_Architecture4
teacher = tf.keras.models.load_model("modelTL_teacher.keras")
teacher.trainable = False
#model = tf.keras.models.load_model("models\model18.keras")
#student = tf.keras.models.load_model("modelTL_student4.keras")
student = tf.keras.models.load_model("modelTL_studentC4.keras")
objectClasses = 10
imagesHeight = 224
imagesWidth = 224

input_shape =(224,224,3)
classes = 10
#Daten aus dem Verzeichnis laden
trainingData = image_dataset_from_directory("Data/train", image_size=(imagesHeight,imagesWidth),batch_size=32,shuffle=True,seed=33)
validationData = image_dataset_from_directory("Data/validation",image_size=(imagesHeight,imagesWidth),batch_size=32)
testData = image_dataset_from_directory("Data/test",image_size=(imagesHeight,imagesWidth),batch_size=32,shuffle=False)
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
    log_dir="logs/runKD_studentC8"
)
#Callbacks setzen
callbacks = [tensorboard_callback,lrReducer,earlyStopping]
#Distiller erzeugen
distiller = Distiller(student=student, teacher=teacher)
#Trainingsparameter uebergeben, dann kompilieren
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=5
)

#Training ausfuehren. Nur die Gewichte des Student werden optimiert.
distiller.fit(
    trainingData,
    validation_data=validationData,
    epochs=35,
    callbacks=callbacks
)
#Modell auf unbekannte Daten testen
distiller.evaluate(testData)
#Modell speichern
student.save("modelTL_studentC8.keras")

#Metadaten der Modelle. Anzahl der Parameter ausgeben
#teacher.summary()
#student.summary()

#ConfusionMatrix berechnen und anzeigen
# y_pred_probs = student.predict(testData)
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