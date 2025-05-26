import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usa backend no interactivo
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class CSVRegressorAuto:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.history = None

    def load_data(self):
        data = pd.read_csv(self.csv_path)

        # Filtramos solo columnas numéricas
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] < 2:
            raise ValueError("The CSV must contain at least two numeric columns for regression.")

        # Por defecto: última columna numérica como Y, el resto como X
        self.y_column = numeric_data.columns[-1]
        self.x_columns = numeric_data.columns[:-1]

        print(f"Using '{self.y_column}' as target variable (y)")
        print(f"Using {list(self.x_columns)} as independent variables (X)")

        self.X = numeric_data[self.x_columns].values
        self.y = numeric_data[[self.y_column]].values

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X = self.scaler_X.fit_transform(self.X)
        self.y = self.scaler_y.fit_transform(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def build_model(self):
        input_dim = self.X.shape[1]
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, epochs=100):
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            verbose=0
        )

    def plot_predictions(self):
        y_pred = self.model.predict(self.X).flatten()
        y_pred_inverse = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_inverse = self.scaler_y.inverse_transform(self.y).flatten()

        plt.figure(figsize=(10, 5))
        plt.plot(y_true_inverse, label="Actual Value", color='blue')
        plt.plot(y_pred_inverse, label="Prediction", color='red')
        plt.xlabel("Sample Index")
        plt.ylabel(self.y_column)
        plt.title("Prediction vs Actual (Inferred)")
        plt.legend()
        plt.grid(True)
        # Guardar en media/grafico.png
        media_path = os.path.join("media", "grafico.png")
        plt.savefig(media_path)
        plt.close()

    def plot_loss(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        # Guardar en media/resultado.png
        media_path = os.path.join("media", "resultado.png")
        plt.savefig(media_path)
        plt.close()

    def generar_informe(self):
        # Define la carpeta de salida 'media' relativa al directorio actual de trabajo.
        # Django usualmente ejecuta desde la raíz del proyecto (donde está manage.py).
        output_directory = "media"
        os.makedirs(output_directory, exist_ok=True) # Asegura que la carpeta 'media' exista

        informe_filename = "informe.txt"
        # Construye la ruta completa al file de informe
        informe_path_final = os.path.join(output_directory, informe_filename)

        try:
            # Predicciones y métricas en test
            y_test_pred = self.model.predict(self.X_test).flatten()
            y_test_pred_inverse = self.scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
            y_test_true_inverse = self.scaler_y.inverse_transform(self.y_test).flatten()

            # Cálculo de métricas
            mse = np.mean((y_test_pred_inverse - y_test_true_inverse) ** 2)
            mae = np.mean(np.abs(y_test_pred_inverse - y_test_true_inverse))

            # Obtener pesos y bias de la última capa
            last_layer = self.model.layers[-1]
            weights, bias = last_layer.get_weights()
            coeficientes = weights.flatten()

            # Ecuación aproximada de la última capa
            eq_str = f"{self.y_column} ≈ " + " + ".join(
                [f"({coef:.3f} * h{i})" for i, coef in enumerate(coeficientes)]
            ) + f" + ({bias[0]:.3f})"
            
            # Guardar 
            with open(informe_path_final, "w", encoding='utf-8') as f:
                f.write("🔍 Regression Report with AI\n")
                f.write("="*40 + "\n")
                f.write(f"CSV File: {self.csv_path}\n\n")
                f.write(f"Target Variable (Y): {self.y_column}\n")
                f.write(f"Independent Variables (X): {list(self.x_columns)}\n\n")
                f.write(f"👉 Approximate learned equation (last layer):\n")
                f.write(eq_str + "\n\n")
                f.write(f"📊 Metrics on test set:\n")
                f.write(f" - MSE (Mean Squared Error): {mse:.4f}\n")
                f.write(f" - MAE (Mean Absolute Error): {mae:.4f}\n")
            
            print(f"✅ Report saved in: {os.path.abspath(informe_path_final)}")

        except Exception as e:
            print(f"❌ Error generating and saving report: {e}")
            print(f"   Attempted to save in: {os.path.abspath(informe_path_final)}")
            # Puedes decidir si relanzar la excepción o no, dependiendo de cómo quieras manejar errores.
            # raise e 
        
        return informe_path_final # Devolver la ruta relativa "media/informe.txt"

