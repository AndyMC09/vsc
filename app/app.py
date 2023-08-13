#importar librerias
import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, render_template, jsonify 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pickle
#para graficos
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#para roc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import io
import os

#Leer archivos del modelo predictivo
modelo_cargado = pickle.load(open('app\model\modelo_predictivo.pkl','rb'))
#Leer archivo csv del dataframe
archivo_dfCSV = 'app\df_bank_balanced.csv'
dfCsv = pd.read_csv(archivo_dfCSV )


app = Flask(__name__)

app.static_folder = 'static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/informacion')
def informacion():
    return render_template('informacion.html')

# Generar Graficos
def generar_histograma(columna):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(dfCsv[columna], bins=20, color='blue', alpha=0.7)
    ax.set_title(f'Histograma de {columna}')
    ax.set_xlabel(columna)
    ax.set_ylabel('Frecuencia')
    ax.grid(True)
    canvas = FigureCanvas(fig)
    buffer = BytesIO()
    canvas.print_png(buffer)
    buffer.seek(0)
    return buffer

def generar_diagrama_pastel(columna):
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = dfCsv[columna].value_counts()
    counts.plot(kind='pie', autopct='%1.1f%%')
    ax.set_title(f'Diagrama de Pastel de {columna}')
    ax.grid(True)
    canvas = FigureCanvas(fig)
    buffer = BytesIO()
    canvas.print_png(buffer)
    buffer.seek(0)
    return buffer

def generar_diagrama_barras(columna):
    fig, ax = plt.subplots(figsize=(8, 9))
    counts = dfCsv[columna].value_counts()
    counts.plot(kind='bar')
    ax.set_title(f'Diagrama de Barras de {columna}')
    ax.set_xlabel(columna)
    ax.set_ylabel('Frecuencia')
    ax.grid(True)
    canvas = FigureCanvas(fig)
    buffer = BytesIO()
    canvas.print_png(buffer)
    buffer.seek(0)
    return buffer


# END Generar Graficos

@app.route('/mostrar_edad')
def mostrar_edad():
    buffer = generar_histograma('Customer_Age')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/mostrar_educacion')
def mostrar_educacion():
    buffer = generar_diagrama_pastel('Education_Level')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/mostrar_estado')
def mostrar_estado():
    buffer = generar_diagrama_barras('Marital_Status')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/mostrar_ingresos')
def mostrar_ingresos():
    buffer = generar_diagrama_pastel('Income_Category')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/mostrar_cantRelaciones')
def mostrar_cantRelaciones():
    buffer = generar_diagrama_pastel('Total_Relationship_Count')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/mostrar_crediticio')
def mostrar_crediticio():
    buffer = generar_histograma('Credit_Limit')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/mostrar_avg')
def mostrar_avg():
    buffer = generar_histograma('Avg_Open_To_Buy')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/mostrar_meses')
def mostrar_meses():
    buffer = generar_diagrama_barras('Months_Inactive_12_mon')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/mostrar_contactos')
def mostrar_contactos():
    buffer = generar_diagrama_pastel('Contacts_Count_12_mon')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/mostrar_montoTransac')
def mostrar_montoTransac():
    buffer = generar_histograma('Total_Trans_Amt')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/mostrar_cantTransac')
def mostrar_cantTransac():
    buffer = generar_histograma('Total_Trans_Ct')
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}


@app.route('/estadistica')
def estadistica():
    buffer = generar_histograma('Customer_Age')  # Generar la imagen por defecto
    imagen_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return render_template('estadistica.html', imagen_base64=imagen_base64)



@app.route('/contacto')
def contacto():
    return render_template('contacto.html')

@app.route('/resultado')
def resultado():
    return render_template('resultado.html')

@app.route('/viewAppPrediction',methods=['POST','GET'])
def viewAppPrediction(): 
    if request.method == 'POST':
        # Obtener los datos del formulario
        datos = [float(x) for x in request.form.values()]
        columData = ["Customer_Age", "Dependent_count", "Total_Relationship_Count",
                     "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit",
                     "Total_Revolving_Bal", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
                     "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
                     "Gender_num", "Marital_Status_num", "Income_Category_num"]

        print("recepcion ", datos)

        #Creando el arreglo, reorganizando sus elementos
        valorDatos = np.array(datos).reshape(-1, 1)
        np.set_printoptions(suppress=True)
        print(valorDatos)

        # se inicializa el escalador
        scaler = MinMaxScaler()

        #se realiza la normalizacion
        valorDatos = scaler.fit_transform(valorDatos)
        print(valorDatos)

        df = pd.DataFrame(data=valorDatos.T, columns=columData)  
        print(df.T.to_string())

        # Crear un diccionario que mapee el nombre de la columna a su valor normalizado
        data_dict = {column: valor for column, valor in zip(columData, valorDatos.flatten())}

        # Crear una lista de tuplas (nombre de la columna, valor normalizado)
        datos_normalizados = [(column, valor) for column, valor in data_dict.items()]

        prediccion = modelo_cargado.predict((df))
        print(prediccion)

        #CURVA ROC
        probs = modelo_cargado.predict_proba(df)
        probs_pos = probs[:, 1]  # Probabilidades de la clase positiva

        # Etiquetas verdaderas
        y_true = label_binarize([1], classes=[0, 1])

        # Calcular la curva ROC
        fpr, tpr, thresholds = roc_curve(y_true, probs_pos)
        roc_auc = auc(fpr, tpr)

        # Crear la gráfica de la curva ROC
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")

        # Convierte la gráfica en un formato embebible en HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        # Renderizar la plantilla y pasar los datos al contexto
        return render_template('resultado.html', datos=datos_normalizados, prediccion=prediccion, img_data=img_data)
    else:
        return render_template('viewAppPrediction.html')


if __name__ == '__main__':
    app.run(host='prediccionesinstitucionesfinancierasgye.azurewebsites.net', port=os.environ.get('PORT', 5000))
