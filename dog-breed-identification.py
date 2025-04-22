import streamlit as st
import numpy as np
import os
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from keras.applications.resnet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Streamlit UI Debugging
st.title("Dog Breed Identification")
st.write("Debug: Streamlit app is running!")

# Load the trained model
MODEL_PATH = "model.keras"
st.write("🔄 Loading model...")
model = load_model(MODEL_PATH)
st.write("✅ Model loaded successfully!")

# Load label encoder
df_labels = pd.read_csv("labels.csv")
breed_dict = list(df_labels['breed'].value_counts().keys()) 
num_breeds = 60
new_list = sorted(breed_dict, reverse=True)[:num_breeds*2+1:2]
df_labels = df_labels.query('breed in @new_list')
encoder = LabelEncoder()
encoder.fit(df_labels["breed"].values)

# Streamlit UI
st.write("Upload an image of a dog, and the model will predict its breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Preprocess image
    im_size = 224
    image = image.resize((im_size, im_size))
    img_array = preprocess_input(np.expand_dims(np.array(image).astype(np.float32), axis=0))
    
    # Predict
    st.write("🔄 Predicting breed...")
    pred_label = model.predict(img_array)
    pred_label = np.argmax(pred_label, axis=1)
    pred_breed = encoder.inverse_transform(pred_label)
    
    # Display result
    st.write("✅ Predicted Breed for this Dog is:", pred_breed[0])