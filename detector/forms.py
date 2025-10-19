from django import forms

class UploadLeafForm(forms.Form):
    crop_image = forms.ImageField(label='Upload Leaf Image')
    crop_type = forms.ChoiceField(
        choices=[('Apple','Apple'),('Corn','Corn'),('Tomato','Tomato'),('Potato','Potato')],
        required=False,
        label='Crop Type (Optional)'
    )
