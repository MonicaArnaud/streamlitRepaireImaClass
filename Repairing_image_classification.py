import streamlit as st
from sklearn import datasets
import numpy as np
import pickle
import os
from PIL import Image

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("维修图片分类器")

st.write("""
## 对维修的图片进行分类，共有四类：电源，控制板，算力板和显卡
""")
st.write("Upload image")

uploaded_img = st.file_uploader('Choose an image to upload...')

#display uploaded image
#img = PIL.Image.create(st.file_uploader.data[0])
#img = uploaded_img.getvalue()
st.image(uploaded_img, caption='Uploaded Image')

# def load_model():
#     with open('Repaire_image_classificationV5.pkl', 'rb') as file:
#         data = pickle.load(file)
#     return data
# #
# data = load_model()
pickle_in = open("Repaire_image_classificationV5.pkl", "rb")
classification_model = pickle.load(pickle_in)

# model = pickle.load(open('Repaire_image_classificationV5.pkl', 'rb'))

pred, pred_idx, probs = data.predict(uploaded_img)


st.write("The machine predicts that the picture belongs to: ", pred)
st.write("The accuracy of the prediction is: ", probs[pred_idx].item())

