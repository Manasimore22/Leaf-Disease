# Importing essential libraries
from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import torch
from torchvision import transforms
from PIL import Image
import io
import os
from .utils.model import ResNet9
from .utils.disease import disease_dic
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Print the current working directory for debugging
print("Current Working Directory:", os.getcwd())

# Initialize Flask app
app = Flask(__name__)

# ------------------------- LOADING THE TRAINED PLANT DISEASE CLASSIFICATION MODEL -------------------------

# Define the list of disease classes
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Path to the trained model file
disease_model_path = os.path.join(os.getcwd(), 'app', 'models', 'plant_disease_model.pth')

# Initialize the model
disease_model = ResNet9(3, len(disease_classes))

# Load the model weights
if os.path.exists(disease_model_path):
    disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu'), weights_only=True))
    disease_model.eval()
else:
    raise FileNotFoundError(f"Model file not found at: {disease_model_path}")

# ------------------------- PREDICTION FUNCTION -------------------------

def predict_image(img, model=disease_model):
    """Predict the disease class for an input image."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure uniform image size
        transforms.ToTensor(),         # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image = Image.open(io.BytesIO(img)).convert('RGB')  # Convert to RGB
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Make prediction
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

# ------------------------- ROUTES -------------------------

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', title='KrishiSutra - Leaf Disease Detection')

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    """Handle disease prediction requests."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)  # Redirect if no file uploaded

        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title='KrishiSutra - Leaf Disease Detection')

        try:
            img = file.read()  # Read the uploaded image
            prediction = predict_image(img)  # Predict disease

            # Get the disease description from the dictionary
            prediction_description = Markup(disease_dic.get(prediction, "No description available."))

            # Render the result page
            return render_template(
                'disease-result.html', 
                prediction=prediction_description, 
                disease_name=prediction, 
                title='KrishiSutra - Leaf Disease Detection'
            )

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('disease.html', title='KrishiSutra - Leaf Disease Detection', error_message=str(e))
    
    return render_template('disease.html', title='KrishiSutra - Leaf Disease Detection')

# ------------------------- MAIN -------------------------

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
