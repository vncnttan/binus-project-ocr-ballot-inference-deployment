import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

def predict(cropped_imgs, model_file, traditional):
    if traditional:
        with open(f'models/{model_file}', 'rb') as file:
            model = pickle.load(file)
    else:
        model = tf.keras.models.load_model(f'models/{model_file}')
    
    tmp_images = np.array(cropped_imgs)

    if traditional:
        tmp_images = tmp_images.reshape(tmp_images.shape[0], -1)
    else:
        tmp_images = np.repeat(tmp_images[..., np.newaxis], 3, -1)
        if model_file == 'ResNet50.h5':
            tmp_images = resnet_preprocess_input(tmp_images)
        elif model_file == 'VGG19.h5':
            tmp_images = vgg19_preprocess_input(tmp_images)

    if traditional:
        return model.predict(tmp_images)

    y_pred = model.predict(tmp_images)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred