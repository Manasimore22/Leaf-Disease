# Importing essential libraries
from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import torch
from torchvision import transforms
from PIL import Image
import io
import os
from utils.model import ResNet9
from utils.disease import disease_dic

# Initialize Flask app
app = Flask(__name__)

# -------------------------LOADING THE TRAINED PLANT DISEASE CLASSIFICATION MODEL ------------------

# Define the list of disease classes
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
                   'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                   'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                   'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
                   'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
                   'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                   'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Load the trained plant disease classification model
# Use os.path.join to make path handling platform-independent
disease_model_path = os.path.join(os.getcwd(), 'app', 'models', 'plant_disease_model.pth')

# Initialize the model
disease_model = ResNet9(3, len(disease_classes))

# Check if the model file exists
if os.path.exists(disease_model_path):
    disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
    disease_model.eval()
else:
    raise FileNotFoundError(f"Model file not found at: {disease_model_path}")

# Define image transformation function
def predict_image(img, model=disease_model):
    transform = transforms.Compose([ 
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

# -------------------------ROUTES ------------------------

# Home page route
@app.route('/')
def home():
    return render_template('index.html', title='KrishiSutra - Leaf Disease Detection')

# Disease detection page
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'POST':
        # Check if the file is in the request
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title='Harvestify - Disease Detection')

        try:
            # Read the image file
            img = file.read()

            # Predict disease
            prediction = predict_image(img)

            # Retrieve the disease description
            prediction_description = Markup(str(disease_dic.get(prediction, "No description available.")))

            # Render the result page
            return render_template('disease-result.html', prediction=prediction_description, title='Harvestify - Disease Detection')

        except Exception as e:
            print(e)
            return render_template('disease.html', title='Harvestify - Disease Detection')
    
    return render_template('disease.html', title='Harvestify - Disease Detection')

# Main block to run the app
if __name__ == '__main__':
    app.run(debug=False)
