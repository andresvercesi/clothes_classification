import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from fastapi import FastAPI



subCategory_model = tf.keras.models.load_model('.\models\modelo_entrenado.h5')
#subCategory_model.summary()

label_mapping = {0: 'Accessories', 1: 'Apparel Set', 2: 'Bags', 3: 'Beauty Accessories',
    4: 'Belts', 5: 'Bottomwear', 6: 'Dress', 7: 'Eyewear', 8: 'Flip Flops', 9: 'Gloves',
    10: 'Headwear', 11: 'Innerwear', 12: 'Jewellery', 13: 'Loungewear and Nightwear',
    14: 'Mufflers', 15: 'Sandal', 16: 'Saree', 17: 'Scarves', 18: 'Shoe Accessories',
    19: 'Shoes', 20: 'Skin', 21: 'Socks', 22: 'Sports Accessories', 23: 'Sports Equipment',
    24: 'Stoles', 25: 'Ties', 26: 'Topwear', 27: 'Umbrellas', 28: 'Wallets', 29: 'Watches'}

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello"}

@app.get("/clothes")
async def clothes(url: str):
    #url = 'https://'+url
    #response = requests.get(url)
    #img = Image.open(BytesIO(response.content))
    img = Image.open(requests.get(url, stream=True).raw)
    new_size = (224, 224)
    img = img.resize(new_size)
    transformed_image_array = img_to_array(img)
    transformed_image_array = preprocess_input(transformed_image_array)
    transformed_image_array = np.expand_dims(transformed_image_array, axis=0)
    print(transformed_image_array.shape)
    #transformed_image_array = np.array(transformed_image_array)
    print(transformed_image_array.shape)
    prediccion = subCategory_model.predict(transformed_image_array)
    
    print(prediccion[0])
    
    etiqueta_predicha = np.argmax(prediccion, axis=1)
    return (label_mapping[etiqueta_predicha[0]])
     
