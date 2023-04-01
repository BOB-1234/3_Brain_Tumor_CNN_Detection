# importing python libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# calling the path of train validation and testing data
test_path = "data/test"

# batch size of 32 is pushed through ML model
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(dtype='float32')
test_generator = test_datagen.flow_from_directory(test_path, batch_size = BATCH_SIZE, target_size = (460,460), class_mode = 'categorical')

## uploading the image path for breast cancer testing
## uncomment the img_path to test out each image for different cases

## has 83.98 % accuracy
#img_path = "Data/test/squamous.cell.carcinoma/000111.png"

## has 88.90 % accuracy
#img_path = "Data/test/large.cell.carcinoma/000110.png"

## has 98.65 % accuracy 
img_path = "Data/test/adenocarcinoma/000122.png"

## has 99.99 % accuracy
#img_path = "Data/test/normal/6.png"

class_names=list(test_generator.class_indices.keys())

img = tf.keras.utils.load_img(img_path, target_size=(460, 460))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# importing the created resnet50 model
from keras.models import load_model
resnet50_model = load_model('ResNet50_model.hdf5')

prediction = resnet50_model.predict(img_array)

#prediction using the uploaded model 
predicted_probabilities = resnet50_model.predict(img_array)
threshold = 0.5

#for loop that prints out the result and accuracy based on image uploaded
for i in range(len(class_names)):
    if predicted_probabilities[0][i] > threshold:
        print("Patient has {} with a {:.2f}% confidence.".format(class_names[i], predicted_probabilities[0][i]*100))
        