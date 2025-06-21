import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

with_mask_folder = 'WithMask'
without_mask_folder = 'WithoutMask'
IMG_SIZE = 128

def load_images_and_masks():
    images = []
    masks = []
    
    for filename in os.listdir(with_mask_folder):
        img_path = os.path.join(with_mask_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Threshold HSV untuk masker biru muda, sesuaikan dengan dataset kamu
        lower = np.array([90, 50, 50])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        mask = mask / 255.0
        mask = mask.astype(np.float32)
        
        images.append(img / 255.0)
        masks.append(mask[..., np.newaxis])
    
    for filename in os.listdir(without_mask_folder):
        img_path = os.path.join(without_mask_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        
        images.append(img / 255.0)
        masks.append(mask)
    
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    return images, masks

def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(input_tensor)
    x = layers.concatenate([x, concat_tensor])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)

    model = models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    X, y = load_images_and_masks()
    print("Loaded images:", X.shape)
    print("Loaded masks:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = build_unet((IMG_SIZE, IMG_SIZE, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, epochs=15, batch_size=16, validation_split=0.1)

    model.save('unet_mask_model.h5')
    print("Model saved as unet_mask_model.h5")
