#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras_tuner')


# In[12]:


import keras
import keras.utils
import keras_tuner
from keras import utils as np_utils
from keras import layers
from keras_tuner.tuners import RandomSearch
from keras.layers import Dropout, BatchNormalization
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow import keras
#from keras.utils import get_custom_objects


# In[15]:


# create nas
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=hp.Int('num_filters', 32, 256, step=32),
                            kernel_size=hp.Choice('kernel_size', values=[3, 5]),
                            activation='relu',padding="SAME",
                            input_shape=(height, width, depth)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hp.Choice('dropout', values=[0.1, 0.2, 0.3])))
    for i in range(hp.Int('num_layers', 1, 10)):
        model.add(layers.Conv2D(filters=hp.Int('filters_' + str(i), 32, 256, step=32),
                                kernel_size=hp.Choice('kernel_size_' + str(i), values=[3, 5]),padding="SAME",
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Choice('dropout_' + str(i), values=[0.1, 0.2, 0.3])))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hp.Int('dense_units', 32, 1024, step=32), activation='relu'))
    model.add(layers.Dropout(hp.Choice('dense_dropout', values=[0.1, 0.2, 0.3])))
    model.add(layers.Dense(output_dim, activation='softmax'))
    model.compile(optimizer='adam',
                  loss= 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

 


# In[ ]:


#define dataset dependent input settings
height, width, depth, output_dim = 28, 28, 1, 10

#defining object for network searching
tuner = RandomSearch(
    build_model,
    objective=(['val_accuracy','val_loss']),max_trials=1,executions_per_trial=1,directory='my_dir',project_name='concentrations'
)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the input data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the input data to have a single channel
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Convert the labels to one-hot encoded vectors
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# network searching
tuner.search(x_train, y_train, epochs=1, batch_size=64,
             validation_split=0.2, validation_steps=10,
             validation_batch_size=16
            )

#getting the best network architecture
print(tuner.get_best_hyperparameters()[0].values)
model = tuner.get_best_models(num_models =1)[0]

#fiting the best network architecture
model.fit(x_train, y_train, epochs=1, batch_size=64, validation_split=0.2, validation_steps=10, validation_batch_size=16)

print("Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test))
# adversarial pattern generation
def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)
    
    gradient = tape.gradient(loss, image)
    
    signed_grad = tf.sign(gradient)
    
    return signed_grad

x_pertubed = x_test
img_rows, img_cols, channels = 28, 28, 1
for i in range(len(x_test)):
    image = x_test[i]
    image_label = y_test[i]
    perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), image_label).numpy()
    adversarial = image + perturbations * 0.1
    x_pertubed[i] = adversarial
print("Base accuracy on pertubes images:", model.evaluate(x=x_pertubed, y=y_test))


# In[18]:


model.summary()


# In[19]:


checkpoint = tf.keras.callbacks.ModelCheckpoint('nas_0.h5', monitor='val_loss',save_best_only=True, verbose=2)


# In[20]:


model.get_weights()[0].dtype


# In[21]:


from tensorflow_model_optimization.quantization.keras import vitis_quantize


# In[29]:


train_calib = np.random.rand( 10,28, 28, 1)


# In[30]:


quantizer = vitis_quantize.VitisQuantizer(model)


# In[31]:


quantized_model = quantizer.quantize_model(calib_dataset = train_calib, weight_bit=8, activation_bit=8)


# In[32]:


quantized_model.compile(loss="MeanAbsoluteError",metrics="MeanAbsoluteError")


# In[33]:


score = quantized_model.evaluate(x_test, y_test,  verbose=0, batch_size=1)


# In[34]:


print(score)


# In[35]:


quantized_model.save('nas_dpu_0.h5')


# In[ ]:




