# uploads/views.py

from django.shortcuts import render, redirect
from django.urls import reverse
from .forms import UploadFileForm
from .models import UploadedFile
from CSVRegressor import CSVRegressorAuto
import os

def upload_file_view(request):
    if request.method == 'POST':
        # Limpiar la carpeta media antes de guardar el nuevo archivo
        media_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media')
        if os.path.exists(media_dir):
            for f in os.listdir(media_dir):
                file_path = os.path.join(media_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            file_path = uploaded_file.archivo.path

            reg = CSVRegressorAuto(file_path)
            reg.load_data()
            reg.build_model()
            reg.train(epochs=200)
            reg.plot_predictions()
            reg.plot_loss()
            
            # 1. Obtener la ruta del informe generado
            informe_file_path_relativa = reg.generar_informe() 
            
            informe_contenido_texto = ""
            # 2. Leer el contenido del archivo
            if informe_file_path_relativa: # Verificar que se obtuvo una ruta
                try:
                    # Asumiendo que la ruta es relativa al directorio raíz del proyecto
                    with open(informe_file_path_relativa, 'r', encoding='utf-8') as f_informe:
                        informe_contenido_texto = f_informe.read()
                except FileNotFoundError:
                    print(f"ADVERTENCIA en la vista: No se pudo encontrar el archivo de informe en {informe_file_path_relativa}")
                except Exception as e:
                    print(f"Error en la vista al leer el archivo de informe {informe_file_path_relativa}: {e}")

            # 3. Añadir el contenido al contexto
            context = {
                'mensaje': 'Archivo procesado y modelo entrenado.', # Puedes añadir otros datos al contexto si los necesitas
                'informe_texto': informe_contenido_texto 
            }
            return render(request, 'uploads/success.html', context)
    else:
        form = UploadFileForm()
    return render(request, 'uploads/upload.html', {'form': form})

def upload_success_view(request):
    # Obtener todos los archivos subidos
    archivos_qs = UploadedFile.objects.all()
    # Crear lista de tuplas (objeto, nombre_solo)
    archivos = []
    for f in archivos_qs:
        nombre_completo = f.archivo.name  # e.g. "uploads/2025/05/22/miarchivo.txt"
        nombre_solo = os.path.basename(nombre_completo)  # -> "miarchivo.txt"
        archivos.append({
            'obj': f,
            'nombre': nombre_solo,
        })

    return render(request, 'sRegression/upload_success.html', {
        'archivos': archivos,
    })



