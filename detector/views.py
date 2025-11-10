import os, json, numpy as np
from django.conf import settings
from django.shortcuts import render
from PIL import Image
import tflite_runtime.interpreter as tflite
import pandas as pd


remedies_df = pd.read_csv(os.path.join(settings.BASE_DIR, 'models', 'remedies.csv'))
remedies_df['disease'] = remedies_df['disease'].str.strip()
remedies = remedies_df.set_index('disease')['remedy'].to_dict()
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'plant_model.tflite')
LABEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'label_map.json')
try:
    with open(LABEL_PATH) as f:
        labels = json.load(f)
    labels = {v: k for k, v in labels.items()}  # reverse mapping

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading ML components (TFLite or JSON): {e}")
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((224,224))
    x = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)

def predict_image(img_path):
    x = preprocess(img_path)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    top = int(np.argmax(preds))
    conf = float(preds[top])
    label = labels[top]
    #print("\n=== Model Predictions ===")
    #for i, p in enumerate(preds):
        #print(f"{labels[i]}: {p:.4f}")
    #print("=========================\n")  
    #print(f"Predicted Label: {label}, Confidence: {conf*100:.2f}%")
    return label, conf  

def upload_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        path = os.path.join(settings.MEDIA_ROOT, image.name)
        with open(path, 'wb+') as dest:
            for chunk in image.chunks():
                dest.write(chunk)
        label, conf = predict_image(path)
        cleaned_label = label.strip()
        remedy = remedies.get(cleaned_label, "No remedy found.")
        # 5. Clean up the temporary file (Good practice)
        try:
            os.remove(path)
        except OSError as e:
            print(f"Error removing temporary file {path}: {e}")
        return render(request, 'result.html', {'label': label, 'confidence': round(conf*100,2), 'remedy': remedy})
        #return render(request, 'result.html', {'label': label, 'confidence': round(conf*100,2)})
    return render(request, 'upload.html')

def home(request):
    return render(request, 'home.html')
#==========================================

import os, json, numpy as np
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image
import tflite_runtime.interpreter as tflite
import pandas as pd

remedies_df = pd.read_csv(os.path.join(settings.BASE_DIR, 'models', 'remedies.csv'))
remedies_df['disease'] = remedies_df['disease'].str.strip()

remedy_dict = remedies_df.set_index('disease')['remedy'].to_dict()
causes_dict = remedies_df.set_index('disease')['causes'].to_dict() if 'causes' in remedies_df.columns else {}
similar_dict = remedies_df.set_index('disease')['similar_diseases'].to_dict() if 'similar_diseases' in remedies_df.columns else {}

MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'plant_model.tflite')
LABEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'label_map.json')

try:
    with open(LABEL_PATH) as f:
        labels = json.load(f)
    labels = {v: k for k, v in labels.items()}  # Reverse mapping if JSON is {label: index}

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading ML model or label map: {e}")


def preprocess(img_path):
    """Resize + normalize image for model"""
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    x = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def predict_image(img_path):
    """Run inference and return (label, confidence)"""
    x = preprocess(img_path)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    top = int(np.argmax(preds))
    conf = float(preds[top])
    label = labels[top]
    return label, conf

def home(request):
    return render(request, 'home.html')


def upload_view(request):
    """Handle image upload, run prediction, and render results"""
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']

        # Save uploaded file to MEDIA_ROOT/uploads/
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        fs = FileSystemStorage(location=upload_dir)
        filename = fs.save(uploaded_file.name, uploaded_file)
        saved_path = os.path.join(upload_dir, filename)
        uploaded_image_url = settings.MEDIA_URL + 'uploads/' + filename

        label, conf = predict_image(saved_path)
        cleaned_label = label.strip()

        remedy = remedy_dict.get(cleaned_label, "No remedy found for this disease.")
        causes = causes_dict.get(cleaned_label, "No causes information available.")
        similar = similar_dict.get(cleaned_label, "")
        similar_list = [s.strip() for s in similar.split(',')] if similar else []

        # os.remove(saved_path)  # Uncomment if you don't need to keep uploaded images

        # Send all data to template
        context = {
            'label': cleaned_label,
            'confidence': round(conf * 100, 2),
            'remedies': remedy,
            'causes': causes,
            'similar_diseases': similar_list,
            'uploaded_image_url': uploaded_image_url
        }

        return render(request, 'result.html', context)

    return render(request, 'upload.html')
