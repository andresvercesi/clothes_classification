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


def open_preprocess_url_image(url):
    #url = 'https://'+url
    #response = requests.get(url)
    #img = Image.open(BytesIO(response.content))
    img = Image.open(requests.get(url, stream=True).raw)
    new_size = (224, 224)
    img = img.resize(new_size)
    transformed_image_array = img_to_array(img)
    del img
    transformed_image_array = preprocess_input(transformed_image_array)
    transformed_image_array = np.expand_dims(transformed_image_array, axis=0)
    return (transformed_image_array)


subCategory_model = tf.keras.models.load_model('subCategoryV1.h5')
articleType_model = tf.keras.models.load_model('articleType_V1.h5')

#subCategory_model.summary()

label_mapping_subcategory = {0: 'Accessories', 1: 'Apparel Set', 2: 'Bags', 3: 'Beauty Accessories',
    4: 'Belts', 5: 'Bottomwear', 6: 'Dress', 7: 'Eyewear', 8: 'Flip Flops', 9: 'Gloves',
    10: 'Headwear', 11: 'Innerwear', 12: 'Jewellery', 13: 'Loungewear and Nightwear',
    14: 'Mufflers', 15: 'Sandal', 16: 'Saree', 17: 'Scarves', 18: 'Shoe Accessories',
    19: 'Shoes', 20: 'Skin', 21: 'Socks', 22: 'Sports Accessories', 23: 'Sports Equipment',
    24: 'Stoles', 25: 'Ties', 26: 'Topwear', 27: 'Umbrellas', 28: 'Wallets', 29: 'Watches'}

label_mapping_articleType = {0: 'Baby Dolls', 1: 'Backpacks', 2: 'Bangle', 3: 'Belts',
    4: 'Blazers', 5: 'Booties', 6: 'Boxers', 7: 'Bra', 8: 'Bracelet', 9: 'Briefs',
    10: 'Camisoles', 11: 'Capris', 12: 'Caps', 13: 'Casual Shoes', 14: 'Clothing Set',
    15: 'Clutches', 16: 'Cufflinks', 17: 'Dresses', 18: 'Duffel Bag', 19: 'Dupatta',
    20: 'Earrings', 21: 'Flats', 22: 'Flip Flops', 23: 'Formal Shoes', 24: 'Gloves',
    25: 'Handbags', 26: 'Hat', 27: 'Heels', 28: 'Innerwear Vests', 29: 'Jackets', 
    30: 'Jeans', 31: 'Jeggings', 32: 'Jewellery Set', 33: 'Jumpsuit', 34: 'Kurta Sets',
    35: 'Kurtas', 36: 'Laptop Bag', 37: 'Leggings', 38: 'Lipstick', 39: 'Lounge Pants',
    40: 'Lounge Tshirts', 41: 'Mens Grooming Kit', 42: 'Messenger Bag', 43: 'Mufflers',
    44: 'Necklace and Chains', 45: 'Night suits', 46: 'Nightdress', 47: 'Pendant',
    48: 'Rain Jacket', 49: 'Rain Trousers', 50: 'Ring', 51: 'Salwar', 52: 'Sandals', 
    53: 'Sarees', 54: 'Scarves', 55: 'Shapewear', 56: 'Shirts', 57: 'Shoe Accessories',
    58: 'Shorts', 59: 'Shrug', 60: 'Skirts', 61: 'Socks', 62: 'Sports Sandals', 
    63: 'Sports Shoes', 64: 'Stockings', 65: 'Sunglasses', 66: 'Sweaters', 
    67: 'Sweatshirts', 68: 'Swimwear', 69: 'Ties', 70: 'Tights', 71: 'Tops',
    72: 'Track Pants', 73: 'Tracksuits', 74: 'Trolley Bag', 75: 'Trousers', 
    76: 'Trunk', 77: 'Tshirts', 78: 'Tunics', 79: 'Waistcoat', 80: 'Wallets',
    81: 'Watches'}

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello"}

@app.get("/subcategory")
async def clothes(url: str):
    results = {}
    
    img = open_preprocess_url_image(url)

    subCategory_inference = subCategory_model.predict(img)
    
    subCategory_prob = subCategory_inference[0]
    
    subcategory_predicted = (np.argmax(subCategory_inference, axis=1))[0]

    print(subCategory_prob[subcategory_predicted])

    results['subCategory'] = label_mapping_subcategory[subcategory_predicted]
    
    return (results)

@app.get("/articleType")
async def clothes(url: str):
    results = {}
    
    img = open_preprocess_url_image(url)

    #subCategory_inference = subCategory_model.predict(img)

    articleType_inference = articleType_model.predict(img)
    
    #subCategory_prob = subCategory_inference[0]
    articleType_prob = articleType_inference[0]
    
    #subcategory_predicted = (np.argmax(subCategory_inference, axis=1))[0]
    articleType_predicted = (np.argmax(articleType_inference, axis=1))[0]

    #print(subCategory_prob[subcategory_predicted])
    print(articleType_prob[articleType_predicted])

    #results['subCategory'] = label_mapping_subcategory[subcategory_predicted]
    results['articleType'] = label_mapping_articleType[articleType_predicted]
    
    return (results)
     
