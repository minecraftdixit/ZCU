#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras


# In[2]:


data = np.random.rand(100,1024,768,3)


# In[3]:


train=data
test=np.random.rand(100,64,48,192)


# input1 = tf.keras.Input(shape=(1024,768,3))
# #x1 = tf.keras.layers.ZeroPadding2D(padding=(2))(input1)
# x1 = tf.keras.layers.Conv2D(128,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(input1)
# x2 = tf.keras.layers.Conv2D(128,3,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x1)
# x3 = tf.keras.layers.Conv2D(128,3,strides=(1,1),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x2)
# x4 = tf.keras.layers.Conv2D(192,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x3)
# x5 = tf.keras.layers.Conv2D(192,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x4)
# model1 = keras.Model(inputs=input1, outputs=x5)

# x5

# x_train, x_test, y_train, y_test=train_test_split(train, test, test_size=0.1)

# model1.compile(optimizer='adagrad',loss="MeanAbsoluteError",metrics="MeanAbsoluteError")
# model1.summary()

# checkpoint = tf.keras.callbacks.ModelCheckpoint('project.h5', monitor='val_loss',
# save_best_only=True, verbose=2)
# model1.fit(x_train,y_train,batch_size=1,epochs=2,validation_split=(1/9),callbacks=[checkpoint])

# model1.get_weights()[0].dtype

# In[4]:


input01 = tf.keras.Input(shape=(1024,768,3))
#x1 = tf.keras.layers.ZeroPadding2D(padding=(2))(input1)
x01 = tf.keras.layers.Conv2D(128,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(input01)
x02 = tf.keras.layers.Conv2D(128,3,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x01)
x03 = tf.keras.layers.Conv2D(128,3,strides=(1,1),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x02)
x04 = tf.keras.layers.Conv2D(192,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x03)
x05 = tf.keras.layers.Conv2D(192,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x04)
x06 = tf.keras.layers.Conv2DTranspose(192,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x05)
x07 = tf.keras.layers.Conv2DTranspose(128,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x06)
x08 = tf.keras.layers.Conv2DTranspose(128,3,strides=(1,1),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x07)
x09 = tf.keras.layers.Conv2DTranspose(128,3,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x08)
x10 = tf.keras.layers.Conv2DTranspose(3,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x09)
model2 = keras.Model(inputs=input01, outputs=x10)


# In[5]:


input01


# In[6]:


x10


# In[7]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(train, train, test_size=0.1)


# In[8]:


model2.compile(optimizer='adam',loss="MeanAbsoluteError",metrics="MeanAbsoluteError")
model2.summary()


# In[112]:


checkpoint2 = tf.keras.callbacks.ModelCheckpoint('project2.h5', monitor='val_loss',
save_best_only=True, verbose=2)
model2.fit(x_train2,y_train2,batch_size=1,epochs=2,validation_split=(1/9),callbacks=[checkpoint2])


# In[ ]:


model2.get_weights()[0].dtype


# In[ ]:


from tensorflow_model_optimization.quantization.keras import vitis_quantize


# In[ ]:


quantizer = vitis_quantize.VitisQuantizer(model2)
quantized_model = quantizer.quantize_model(calib_dataset = train, weight_bit=8, activation_bit=8)


# In[ ]:


quantized_model.compile(loss="MeanAbsoluteError",metrics="MeanAbsoluteError")

score = quantized_model.evaluate(x_test2, y_test2,  verbose=0, batch_size=1)
print(score)


# In[ ]:


quantized_model.save('custom_dpu_2.h5')

