# sRegression
Simple Regression Python Django Project , where you can do a regression to a csv file using AI Tensor Flow
Como bajar django en mac en entorno virtual

Primero 

python3 -m venv venv

Para activar el entorno virtual en IOS se ejecuta

source venv/bin/activate 

Luego del entorno virtual ejecutar

pip install django

Para crear el archivo requeriments

pip freeze > requirements.txt

Como crear el proyecto en Django

django-admin startproject <nombre del proyecto>

Para crear un proyecto con mis propias configuraciones

django-admin startproject config . 

Creara la estructura config del proyecto.

Para hace correr el servidor

python3 manage.py runserver

Para crear nuestra applicacion ejecutamos

python3 manage.py startapp alumnos

Para dockerizar 

docker compose up --build

docker compose up -d --build

docker ps

docker compose logs -f