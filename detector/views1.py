# previous version of detector/views.py
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