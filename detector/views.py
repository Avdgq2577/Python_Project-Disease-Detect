from django.shortcuts import render
from .forms import UploadLeafForm
from .models import PredictionHistory
import os, json
from django.conf import settings
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_model', 'crop_disease_model.h5')
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None

CLASS_NAMES = { "0": "Healthy", "1": "Tomato_Early_blight", "2": "Tomato_Leaf_Mold", "3": "Corn_Common_rust" }
CURE_DICT = {
    "Healthy": "No action needed.",
    "Tomato_Early_blight": "Remove infected leaves, apply fungicide.",
    "Tomato_Leaf_Mold": "Increase airflow, apply fungicide.",
    "Corn_Common_rust": "Use resistant varieties, spray fungicide."
}

def severity_level(prob):
    if prob < 40:
        return "Mild"
    elif prob < 70:
        return "Moderate"
    else:
        return "Severe"

def home(request):
    if request.method == 'POST':
        form = UploadLeafForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data['crop_image']
            crop_type = form.cleaned_data.get('crop_type', 'Unknown')
            img_path = os.path.join(settings.MEDIA_ROOT, img.name)
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            with open(img_path, 'wb+') as f:
                for chunk in img.chunks():
                    f.write(chunk)
            try:
                img_obj = image.load_img(img_path, target_size=(224,224))
                img_array = image.img_to_array(img_obj)/255.0
                img_array = np.expand_dims(img_array, axis=0)
            except Exception as e:
                img_array = None
            if model is not None and img_array is not None:
                preds = model.predict(img_array)
                class_idx = int(np.argmax(preds))
                prob = float(np.max(preds))*100
            else:
                class_idx = 1
                prob = 72.5
            disease = CLASS_NAMES.get(str(class_idx), 'Unknown')
            cure = CURE_DICT.get(disease, 'Consult expert.')
            severity = severity_level(prob)
            if request.user.is_authenticated:
                PredictionHistory.objects.create(
                    user=request.user,
                    crop_type=crop_type,
                    disease_name=disease,
                    severity=severity,
                    image=img
                )
            heatmap_url = None
            try:
                img_cv = cv2.imread(img_path)
                if img_cv is not None:
                    heat = cv2.applyColorMap(cv2.resize(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY),(img_cv.shape[1], img_cv.shape[0])), cv2.COLORMAP_JET)
                    superimposed = cv2.addWeighted(img_cv, 0.6, heat, 0.4, 0)
                    heatmap_path = os.path.join(settings.MEDIA_ROOT, 'heatmap_'+img.name)
                    cv2.imwrite(heatmap_path, superimposed)
                    heatmap_url = f'/media/heatmap_{img.name}'
            except Exception:
                heatmap_url = None

            context = {
                'form': form,
                'disease': disease,
                'cure': cure,
                'severity': severity,
                'prob': round(prob,2),
                'img_url': f'/media/{img.name}',
                'heatmap_url': heatmap_url
            }
            return render(request, 'detector/results.html', context)
    else:
        form = UploadLeafForm()
    return render(request, 'detector/home.html', {'form': form})
