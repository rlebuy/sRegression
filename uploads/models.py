# uploads/models.py

from django.db import models

class UploadedFile(models.Model):
    archivo = models.FileField(
        upload_to='uploads/%Y/%m/%d/',
        verbose_name='Archivo'
    )
    subido_el = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.archivo.name
