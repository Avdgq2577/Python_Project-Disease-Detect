# CropDetect - Crop Disease Detection (Enhanced Version)

This is a full-featured Django project for **CropDetection**:
- Crop disease detection using a pre-trained CNN model
- Similar disease suggestion
- Severity estimation
- Multi-crop-leaf support
- Recommendation database and prediction history

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
