# uploads/forms.py

from django import forms
from .models import UploadedFile

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['archivo']
        widgets = {
            'archivo': forms.FileInput(attrs={
                'class': 'custom-file-input',
                'accept': '.csv'  # Añade esto
            }),
        }
    # Si no usas ModelForm, sería algo como:
    # archivo = forms.FileField(widget=forms.FileInput(attrs={'class': 'custom-file-input', 'accept': '.csv'}))
