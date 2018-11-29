import tensorflow
import keras
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout, Flatten
from keras.models import load_model
import sys
from keras.layers import Conv2D, MaxPooling2D 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


test_data=sys.argv[3]
img_rows, img_cols=120,120
input_shape=(3, img_rows, img_cols) 
epochs=5
batch_size=10 
nb_train_samples=600 
nb_validation_samples=120 
nb_test_samples=120\

Test_generator = Datagen.flow_from_directory(test_data,
                                        target_size=(120, 120),
                                        batch_size = 5,
                                        class_mode='categorical',
                                        shuffle=False)

scores = model.evaluate_generator(Test_generator, nb_test_samples )
print("Точность на тестовых данных: %.2f%%" % (scores[1]*100))
