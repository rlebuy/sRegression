# uploads/urls.py

from django.urls import path
from . import views

app_name = 'uploads'
urlpatterns = [
    path('', views.upload_file_view, name='upload'),
    path('success/', views.upload_success_view, name='upload_success'),
]
