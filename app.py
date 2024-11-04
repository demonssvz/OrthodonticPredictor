import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
 
model_filename = 'model1.pkl'

model_filename2 = 'model.pkl'

if not os.path.exists(model_filename):
    st.error("Model file 'model.pkl' not found. Please upload the model file to proceed.")
else:
    # Load the pre-trained model
    model = joblib.load(model_filename)
    model2 = joblib.load(model_filename2)
 
 
resnet_model = models.resnet18(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove the final layer
resnet_model.eval()
 
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 

st.title("P-value Predictor for Dental X-ray Images")
 
preimage_number = st.number_input("Enter Preimage Number:", min_value=0)
 
uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file: 
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
 
    x, y, w, h = 400, 200, 100, 100
    image_array = np.array(image)
    cropped_image = image_array[y:y+h, x:x+w]
 
    st.image(cropped_image, caption='Cropped Region', use_column_width=True)
 
    cropped_image = Image.fromarray(cropped_image)   
    input_tensor = preprocess(cropped_image)
    input_batch = input_tensor.unsqueeze(0)   

    # Extract features using the ResNet18 model
    with torch.no_grad():
        features = resnet_model(input_batch)

    # Flatten the features
    features_flat = features.squeeze().numpy()
    st.subheader("First Few Features of the Extracted Feature Vector")
    st.write(features_flat[:10])
 
    st.subheader("Extracted Feature Vector (Line Chart)")
    st.line_chart(features_flat)
 
if st.button("Predict P-value"):
    # Create a DataFrame for the prediction
    new_data = pd.DataFrame({
        'preimagenumber': [preimage_number]  # Use the correct feature name
    })
 
    prediction = model.predict(new_data)
    prediction2 = model2.predict(new_data)
 
    st.write(f"Predicted Post-treatment length is : {prediction[0]:.4f} and the p-value is  {prediction2[0]:.4f} ")
