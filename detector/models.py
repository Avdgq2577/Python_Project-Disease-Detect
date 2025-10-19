from django.db import models
from django.contrib.auth.models import User

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    crop_type = models.CharField(max_length=50)
    disease_name = models.CharField(max_length=100)
    severity = models.CharField(max_length=20)
    image = models.ImageField(upload_to='uploads/')
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.crop_type} - {self.disease_name} ({self.severity})"
