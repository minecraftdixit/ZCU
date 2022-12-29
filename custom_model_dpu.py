#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras


# In[5]:


data = np.random.rand(10,1024,768,3)
test=np.random.rand(10,64,48,192)
train=data
#test=np.random.rand(100,64,48,192)


# In[6]:


input1 = tf.keras.Input(shape=(1024,768,3))
#x1 = tf.keras.layers.ZeroPadding2D(padding=(2))(input1)
x1 = tf.keras.layers.Conv2D(128,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(input1)
x2 = tf.keras.layers.Conv2D(128,3,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x1)
x3 = tf.keras.layers.Conv2D(128,3,strides=(1,1),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x2)
x4 = tf.keras.layers.Conv2D(192,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x3)
x5 = tf.keras.layers.Conv2D(192,5,strides=(2,2),padding='same',dilation_rate=(1, 1),groups=1,activation=None,use_bias=True)(x4)
model1 = keras.Model(inputs=input1, outputs=x5)


# In[7]:


x5
#model1 : input to x3


# In[7]:


x_train, x_test, y_train, y_test=train_test_split(train, test, test_size=0.1)


# In[8]:


model1.compile(optimizer='adagrad',loss="MeanAbsoluteError",metrics="MeanAbsoluteError")
model1.summary()


# In[9]:


checkpoint = tf.keras.callbacks.ModelCheckpoint('project_0.h5', monitor='val_loss',
save_best_only=True, verbose=2)
model1.fit(x_train,y_train,batch_size=1,epochs=2,validation_split=(1/9),callbacks=[checkpoint])


# In[10]:


model1.get_weights()[0].dtype


# In[11]:


from tensorflow_model_optimization.quantization.keras import vitis_quantize


# In[ ]:


train_calib = np.random.rand(100,1024,768,3)


# In[ ]:


quantizer = vitis_quantize.VitisQuantizer(model1)
quantized_model = quantizer.quantize_model(calib_dataset = train_calib, weight_bit=8, activation_bit=8)


# In[11]:


quantized_model.compile(loss="MeanAbsoluteError",metrics="MeanAbsoluteError")

score = quantized_model.evaluate(x_test, y_test,  verbose=0, batch_size=1)
print(score)


# In[12]:


quantized_model.save('custom_dpu_0.h5')


# In[14]:


get_ipython().system('vai_c_tensorflow2      --model ./custom_dpu_0.h5      --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json      --output_dir .      --net_name custom_dpu_compiled_0')

