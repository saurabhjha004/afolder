import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn

# Define the CNN model (same as used in training)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the trained model
num_classes = 39  # Update to match your dataset
model = CNNModel(num_classes=num_classes)
model.load_state_dict(torch.load("plant_disease_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define class names (matching your dataset structure)
class_names = [
    "Apple__Apple_scab", "Apple__Black_rot", "Apple__Cedar_apple_rust", "Apple__healthy",
    "Background_without_leaves", "Blueberry__healthy", "Cherry__healthy", "Cherry__Powdery_mildew",
    "Corn__Cercospora_leaf_spot Gray_leaf_spot", "Corn__Common_rust", "Corn__healthy",
    "Corn__Northern_Leaf_Blight", "Grape__Black_rot", "Grape__Esca_(Black_Measles)",
    "Grape__healthy", "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange__Haunglongbing_(Citrus_greening)",
    "Peach__Bacterial_spot", "Peach__healthy", "Pepper,_bell__Bacterial_spot", "Pepper,_bell__healthy",
    "Potato__Early_blight", "Potato__healthy", "Potato__Late_blight", "Raspberry__healthy",
    "Soybean__healthy", "Squash__Powdery_mildew", "Strawberry__healthy", "Strawberry__Leaf_scorch",
    "Tomato__Bacterial_spot", "Tomato__Early_blight", "Tomato__healthy", "Tomato__Late_blight",
    "Tomato__Leaf_Mold", "Tomato__Septoria_leaf_spot", "Tomato__Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot", "Tomato__Tomato_mosaic_virus", "Tomato__Tomato_Yellow_Leaf_Curl_Virus"
]

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Streamlit UI
st.title("Plant Disease Detection")
st.write("Upload an image of a plant leaf, and the model will predict its disease status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_index = torch.argmax(prediction, dim=1).item()
            predicted_class = class_names[predicted_index]

        # Determine if the leaf is healthy or diseased
        if "healthy" in predicted_class:
            st.success("The leaf is **Healthy**.")
        else:
            st.error(f"The leaf is **Diseased**: {predicted_class.replace('__', ' ')}")
    except Exception as e:
        st.error(f"Error: {e}")
