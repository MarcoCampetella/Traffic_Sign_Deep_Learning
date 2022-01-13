#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from imgaug import augmenters as iaa


# In[56]:


#def concatenateLista(lista):
#    new_lista = []
#    for i in range(len(lista[:,0,0,0])):
#        new_lista += lista[i,...]
#    return new_lista


# In[6]:


data = []
labels = []
classes = 43
cur_path = os.getcwd()
#print(cur_path)


# In[9]:


seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order
#images_aug = seq(images=images)


# In[10]:


#Retrieving the images and their labels 
for i in range(classes):
    path = os.path.join(cur_path,'archive/Train',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            #print(path + '/'+ a)
            image = Image.open(path + '/'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            #sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")


# In[11]:


#Converting lists into numpy arrays
#data = np.array(data)
#labels = np.array(labels)


# In[12]:


#print(data.shape, labels.shape)


# In[40]:


#unique,counts = np.unique(labels,return_counts=True)
#dizionario = dict(zip(unique,counts))
#massimo = np.max(counts)
#massimo = max(dizionario, key=dizionario.get)
#minimo = min(dizionario, key=dizionario.get)
#print((massimo))
#print((minimo))
#print(dizionario)


# In[41]:


#print(len(data[labels==0,0,0,0]))


# In[64]:


#data_prova = []
#for i in range(10):
#    data_aug = seq(images=data[labels==2,:,:,:])
#    data_prova.append(data_aug)
#print(np.array(data_prova).shape)


# In[68]:


#data_new = []
#labels_new = []
#for i in range(classes):
#    number_of_ele = len(data[labels==i,0,0,0])
#    #print(massimo,number_of_ele)
#    if (massimo // number_of_ele) > 1:
#        n_aug = massimo // number_of_ele
##        data_app = data[labels==i,:,:,:]
#        #print(data_app.shape,n_aug)
#        data_new.append(data[labels==i,:,:,:])
#        print(np.array(data_new).shape)
#        for i in range(n_aug-1):
#            data_aug = seq(images=data[labels==i,:,:,:])
#            data_new.append(data_aug)
#    else:
#        data_new.append(data[labels==i,:,:,:])
#        print(np.array(data_new).shape)
#        #print(data_app.shape,massimo // number_of_ele)
#    #print(number_of_ele)


# In[67]:


#data_new03 = np.array(data_new)
#print(data_new03.shape)


# In[61]:


#data_new02 = concatenateLista(data_new)
#data_new02 = np.reshape(np.array(data_new),(176*210,30,30,3))
#print(data_new02.shape)


# In[58]:


#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


# In[17]:


#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))


# In[18]:


#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")


# In[19]:


#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[24]:


#testing accuracy on test dataset
from sklearn.metrics import accuracy_score

y_test = pd.read_csv('Test.csv')

#y_test["Path"] = y_test["Path"].str.cat("archive", sep="/")
#print(y_test["Path"][0:10])

labels = y_test["ClassId"].values
imgs = y_test["Path"].values


data=[]

for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))

X_test=np.array(data)

pred = model.predict_classes(X_test)

#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))


# In[ ]:




