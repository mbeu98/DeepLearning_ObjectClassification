import tensorflow as tf
from keras.utils import image_dataset_from_directory
from architectures import Architecture18
import numpy as np

objectClasses = 10
imagesHeight = 224
imagesWidth = 224


trainingData = image_dataset_from_directory("Data/train", image_size=(imagesHeight,imagesWidth),batch_size=32,shuffle=True,seed=33)
validationData = image_dataset_from_directory("Data/validation",image_size=(imagesHeight,imagesWidth),batch_size=32)
testData = image_dataset_from_directory("Data/test",image_size=(imagesHeight,imagesWidth),batch_size=32,shuffle=False)
class_names = trainingData.class_names
print(class_names)

trainingData = trainingData.map(lambda x, y:(x / 255.0, y))
validationData = validationData.map(lambda x, y:(x / 255.0, y))
testData = testData.map(lambda x, y:(x / 255.0, y))

trainingData = trainingData.prefetch(tf.data.AUTOTUNE)
validationData = validationData.prefetch(tf.data.AUTOTUNE)
testData = testData.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.load_model("models\model18.keras")

#model = Architecture18.getModel()

# lrReducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3)
# earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#     log_dir="logs/run30"
# )
#
# model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'],optimizer="adam")
# history = model.fit(trainingData, epochs=70,validation_data=validationData, callbacks=[earlyStopping,tensorboard_callback,lrReducer])
# model.save("model30.keras")
#
# test_loss, test_acc = model.evaluate(testData,verbose=2)
# print('Test accuracy:', test_acc)

img_path = r"C:\Users\mikab\PycharmProjects\DeepLearning\Data\test\Hoopoe(Wiedehopf)\Hoopoe_817.jpg"

img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

print(prediction)



pred_class = np.argmax(prediction)
print("Klasse:", class_names[pred_class])