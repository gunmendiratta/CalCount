import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import json
import os

# Ensure gdown is installed
try:
    import gdown
except ImportError:
    os.system('pip install gdown')
    import gdown

model_path = "Food_model.keras"
if not os.path.exists(model_path):
    file_id = "1sRanYcwpDe0mD2uqv4tYjO7L4Gfhyd7K"
    url = f"https://drive.google.com/file/d/1sRanYcwpDe0mD2uqv4tYjO7L4Gfhyd7K/view?usp=share_link"
    gdown.download(url, model_path, quiet=False)
 # Make sure this matches your saved model filename
CLASSES_TXT_PATH = 'classes.txt'
CALORIE_DB_PATH = os.path.join('Cal.json') # Or 'Cal.json' if you renamed it

# --- 1. Load Model, Class Names, and Calorie Data (Cached for performance) ---
# Use st.cache_resource to load heavy resources like models only once
@st.cache_resource
def load_all_resources():
    model = None
    sorted_class_names = []
    calorie_df = pd.DataFrame()

    # Load Model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")

    # Load Class Names
    try:
        with open(CLASSES_TXT_PATH, 'r') as f:
            class_names = [line.strip() for line in f]
        sorted_class_names = sorted(class_names)
    except Exception as e:
        st.error(f"Error loading class names from {CLASSES_TXT_PATH}: {e}")

    # Load Calorie Database
    try:
        with open(CALORIE_DB_PATH, 'r') as f:
            calorie_data = json.load(f)
        calorie_df = pd.DataFrame.from_dict(calorie_data, orient='index')
    except Exception as e:
        st.error(f"Error loading calorie data from {CALORIE_DB_PATH}: {e}")

    return model, sorted_class_names, calorie_df

model, sorted_class_names, calorie_df = load_all_resources()


# --- 2. Image Preprocessing Function ---
def preprocess_image(uploaded_file, target_size=(224, 224)):
    img = image.load_img(uploaded_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize
    return img_array

# --- 3. Prediction Function ---
def predict_food_item(img_array, model, sorted_class_names):
    if model is None or not sorted_class_names:
        return "N/A", 0.0 # Return default if resources aren't loaded

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_food_name = sorted_class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    return predicted_food_name, confidence

# --- 4. Calorie Info Function ---
def get_calorie_info(food_name, calorie_df):
    if calorie_df.empty:
        return {"error": "Calorie database not loaded or empty."}
    try:
        info = calorie_df.loc[food_name].to_dict()
        return info
    except KeyError:
        return {"error": f"Calorie information not found for '{food_name}' in database."}

# --- Streamlit App Layout ---

st.set_page_config(page_title="Food Calorie Counter", layout="centered")

st.title("🍔 Food Calorie Counter 🍎")
st.write("Upload an image of food to estimate its type and calorie content.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Analyzing...")

    # Preprocess and Predict
    img_array = preprocess_image(uploaded_file)
    predicted_food, confidence = predict_food_item(img_array, model, sorted_class_names)

    # Get Calorie Info
    calorie_info = get_calorie_info(predicted_food, calorie_df)

    st.subheader("Analysis Result:")
    st.write(f"**Predicted Food:** {predicted_food} (Confidence: {confidence*100:.2f}%)")

    st.subheader("Calorie Information:")
    if "error" in calorie_info:
        st.error(calorie_info["error"])
    else:
        for key, value in calorie_info.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

    st.markdown("---")
    st.write("Upload a new image to analyze another food item.")