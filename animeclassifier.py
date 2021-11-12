#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[2]:


import cv2


# In[3]:


from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


# In[4]:


img = image.load_img('C:\\basedata\\val\\luffy\\4.jpg')


# In[5]:


plt.imshow(img)


# In[6]:


cv2.imread('C:\\basedata\\val\\luffy\\4.jpg').shape


# In[7]:


train = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)
validation = ImageDataGenerator(rescale=1./255)


# In[8]:


train_dataset = train.flow_from_directory('C:\\basedata\\train',
                                         target_size=(200,200),
                                         batch_size=3,
                                         class_mode = 'binary')
validation_dataset = validation.flow_from_directory('C:\\basedata\\val',
                                         target_size=(200,200),
                                         batch_size=3,
                                         class_mode = 'binary')


# In[9]:


train_dataset.class_indices


# In[10]:


train_dataset.classes


# In[11]:


cnn = models.Sequential([
    layers.Conv2D(64,(3,3),activation='relu',input_shape=(200,200,3)),
    layers.MaxPool2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPool2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPool2D(2,2),
    layers.GlobalMaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])


# In[12]:


from tensorflow.keras.optimizers import RMSprop
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[13]:


model_fit = cnn.fit(train_dataset,epochs=20,validation_data=validation_dataset)


# In[14]:


import numpy as np
dir_path = 'C:\\basedata\\test'
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    X=image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    val = cnn.predict(images)
    if val == 1:
        print('Its Naruto Uzumaki from Naruto Shippuden')        
    else:
        print('Its Monkey D. Luffy from One Piece')

