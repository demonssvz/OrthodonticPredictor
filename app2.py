import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# Define model filenames
model_filename1 = 'model1.pkl'
model_filename2 = 'model.pkl'

# Check if both model files exist
if not os.path.exists(model_filename1) or not os.path.exists(model_filename2):
    st.error("Model files not found. Please upload 'model1.pkl' and 'model.pkl' to proceed.")
else:
    # Load the pre-trained models
    model1 = joblib.load(model_filename1)
    model2 = joblib.load(model_filename2)

# Load ResNet model for feature extraction
resnet_model = models.resnet18(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove the final layer
resnet_model.eval()

# Define preprocessing steps for the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit app title and description
st.title("P-value Predictor for Dental X-ray Images")

# Input for the preimage number
preimage_number = st.number_input("Enter Preimage Number:", min_value=0)

# File uploader for the dental X-ray image
uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file: 
    # Load and display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Define coordinates for cropping
    x, y, w, h = 400, 200, 100, 100
    image_array = np.array(image)
    cropped_image = image_array[y:y+h, x:x+w]

    # Display cropped region
    st.image(cropped_image, caption='Cropped Region', use_column_width=True)

    # Preprocess cropped region and extract features
    cropped_image = Image.fromarray(cropped_image)
    input_tensor = preprocess(cropped_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Extract features using the ResNet18 model
    with torch.no_grad():
        features = resnet_model(input_batch)

    # Flatten the features for visualization
    features_flat = features.squeeze().numpy()
    st.subheader("First Few Features of the Extracted Feature Vector")
    st.write(features_flat[:10])

    # Plot the extracted feature vector
    st.subheader("Extracted Feature Vector (Line Chart)")
    st.line_chart(features_flat)

# Button to predict the p-value and post-treatment length
if st.button("Predict P-value"):
    # Create a DataFrame for prediction, using the exact feature name as in the training data
    new_data = pd.DataFrame({
        'PreimageNumber': [preimage_number]  # Ensure name matches training feature name
    })

    try:
        # Make predictions using both models
        prediction_length = model1.predict(new_data)
        prediction_pvalue = model2.predict(new_data)

        # Display predictions
        st.write(f"Predicted Post-treatment length is: {prediction_length[0]:.4f}")
        st.write(f"Predicted p-value is: {prediction_pvalue[0]:.4f}")
    except ValueError as e:
        st.error(f"An error occurred: {e}")
        st.write("Please check that the input data aligns with model expectations.")
