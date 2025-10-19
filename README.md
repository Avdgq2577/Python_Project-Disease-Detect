# CropDetect - Crop Disease Detection (Enhanced Version)

This is a full-featured Django project skeleton for **CropDetection**:
- Crop disease detection using a pre-trained CNN model (placeholder included)
- Grad-CAM visualization for explainability
- Severity estimation
- Multi-crop support
- Recommendation database and prediction history
- Analytics dashboard placeholders

## How to run locally
1. Create virtual env:
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows

2. Install requirements:
   pip install -r requirements.txt

3. Apply migrations:
   python manage.py migrate

4. Create superuser (optional):
   python manage.py createsuperuser

5. Run server:
   python manage.py runserver

6. Open: http://127.0.0.1:8000/

Note: The `ml_model/crop_disease_model.h5` is a small placeholder file. Replace with your trained model for actual predictions.
