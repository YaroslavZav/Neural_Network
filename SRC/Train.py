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

train_data=sys.argv[2]
test_data=sys.argv[3]
val_data=sys.argv[4]

img_rows, img_cols=120,120
input_shape=(3, img_rows, img_cols) 
epochs=5
batch_size=10 
nb_train_samples=600 
nb_validation_samples=120 
nb_test_samples=120


model = Sequential()

model.add(Conv2D(35, (3,3), activation = 'relu', input_shape=(120, 120, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(25, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(60))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('sigmoid'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
Datagen = ImageDataGenerator(rescale=1./255)
Train_generator = Datagen.flow_from_directory(train_data,
                                        target_size=(120, 120),
                                        batch_size = 10,
                                        class_mode='categorical',
                                        shuffle=False)
Test_generator = Datagen.flow_from_directory(test_data,
                                        target_size=(120, 120),
                                        batch_size = 10,
                                        class_mode='categorical',
                                        shuffle=False)
Val_generator = Datagen.flow_from_directory(val_data,
                                        target_size=(120, 120),
                                        batch_size = 10,
                                        class_mode='categorical',
                                        shuffle=False)
                                       
 model.fit_generator(Train_generator,
                    steps_per_epoch=nb_train_samples ,
                    epochs=epochs,
                    validation_data=Val_generator,
                    validation_steps=nb_validation_samples )
 model.save(sys.argv[1])
 
