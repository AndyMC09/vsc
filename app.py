#importar librerias
import matplotlib
matplotlib.use("Agg")
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



#Leer archivos del modelo predictivo
modelo_cargado = pickle.load(open('app/model/modelo_predictivo.pkl','rb'))
#Leer archivo csv del dataframe
archivo_dfCSV = 'app/df_bank_balanced.csv'
dfCsv = pd.read_csv(archivo_dfCSV )

#CREAR INSTANCIA DE UNA APLICACIÓN FLASK
app = Flask(__name__)

app.static_folder = 'static'

#DEFINE RUTA Y FUNCIÓN DE VISTA
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
        

        scaler = MinMaxScaler()
        

        #se realiza la normalizacion
        valorDatos = scaler.fit_transform(valorDatos)
        print("normali",valorDatos)
        
        df = pd.DataFrame(data=valorDatos.T, columns=columData)  
        print(df)


        #NEW
        #OBTENER DATOS
        edad = float(request.form.get('edad'))
        cantDependientes = int(request.form.get('cantDependientes'))
        cantRelaciones = int(request.form.get('cantRelaciones'))
        cantInactividad = int(request.form.get('cantInactividad'))
        cantContactos = int(request.form.get('cantContactos'))
        limCrediticio = float(request.form.get('limCrediticio'))
        saldoRevolvente = float(request.form.get('saldoRevolvente'))
        cambioTotalMonto = float(request.form.get('cambioTotalMonto'))
        montoTransac = float(request.form.get('montoTransac'))
        cantTransac = int(request.form.get('cantTransac'))
        cambioTotalCantidad = float(request.form.get('cambioTotalCantidad'))
        promUtilizacion = float(request.form.get('promUtilizacion'))
        genero = int(request.form.get('genero'))
        estadoCivil = int(request.form.get('estadoCivil'))
        catIngresos = int(request.form.get('catIngresos'))

        #PRESENTAR DATOS
        print("Edad:", edad)
        print("Cantidad de Dependientes:", cantDependientes)
        print("Cantidad de Relaciones:", cantRelaciones)
        print("Meses de Inactividad:", cantInactividad)
        print("Cantidad de Contactos:", cantContactos)
        print("Límite Crediticio:", limCrediticio)
        print("Saldo Total Revolvente:", saldoRevolvente)
        print("Cambio Total de Monto:", cambioTotalMonto)
        print("Monto Total de Transacciones:", montoTransac)
        print("Cantidad Total de Transacciones:", cantTransac)
        print("Cambio Total de Cantidad:", cambioTotalCantidad)
        print("Promedio de Relación de Utilización:", promUtilizacion)
        print("Género:", genero)
        print("Estado Civil:", estadoCivil)
        print("Categoría de Ingresos:", catIngresos)
        
        #CALCULO DATOS
        calcEdad =(edad-26)/(68-26)
        calcDependientes =(cantDependientes -0)/(5-0)
        calcRelaciones = (cantRelaciones - 1)/(6-1)
        calcInactividad = (cantInactividad -0)/(6-0)
        calcContactos = (cantContactos - 0)/(6-0)
        calcLimCrediticio =(limCrediticio - 1438.3)/(34516-1438.3)
        calcSaldoRevolvente = (saldoRevolvente -0)/(2517-0)
        calcCambioTotalmonto =(cambioTotalMonto - 0)/(3.355-0)
        calcMontoTransac=(montoTransac -  510)/(17995-510)
        calcCantTransac =(cantTransac - 10)/(139-10)
        calcCambioTotalCantidad =(cambioTotalCantidad -0)/(3-0)
        calcPromUtilizacion =(promUtilizacion -0)/(0.999-0)
        calcGenero=(genero - 0)/(1-0)
        calcEstadoCivil=(estadoCivil-0)/(3-0)
        calcCatIngresos=(catIngresos - 0)/(5-0)
        
        #LISTAR DATOS
        listaDatos =[calcEdad,calcDependientes,calcRelaciones,calcInactividad,
                     calcContactos,calcLimCrediticio,calcSaldoRevolvente,
                      calcCambioTotalmonto,calcMontoTransac,calcCantTransac,
                       calcCambioTotalCantidad,calcPromUtilizacion,calcGenero,
                        calcEstadoCivil, calcCatIngresos]
        
        print("NEW",listaDatos)
        #CREA UN ARREGLO CON LA LISTA
        dataNEW=np.array([listaDatos])
        print("NEW",dataNEW)
        #CREA DATAFRAME APARTIR DEL ARREGLO E INCLUYE LAS COLUMNAS 
        dataDF = pd.DataFrame(data=dataNEW, columns=columData)  
        print("NEW",dataDF)


        #REALIZA LA PREDICCIÓN
        prediccion = modelo_cargado.predict((dataDF))
        print(prediccion)
        

        # Crear un diccionario que mapee el nombre de la columna a su valor normalizado
        data_dict = {column: valor for column, valor in zip(columData, dataNEW.flatten())}

        # Crear una lista de tuplas (nombre de la columna, valor normalizado)
        datos_normalizados = [(column, valor) for column, valor in data_dict.items()]

        # Renderizar la plantilla y pasar los datos al contexto
        return render_template('resultado.html', datos=datos_normalizados, prediccion=prediccion)
    else:
        return render_template('viewAppPrediction.html')


"""@app.route('/viewAppPrediction',methods=['POST','GET'])
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
        print("normali",valorDatos)
        
        df = pd.DataFrame(data=valorDatos.T, columns=columData)  
        print(df)

        prediccion = modelo_cargado.predict((df))
        print(prediccion)

        # Crear un diccionario que mapee el nombre de la columna a su valor normalizado
        data_dict = {column: valor for column, valor in zip(columData, valorDatos.flatten())}

        # Crear una lista de tuplas (nombre de la columna, valor normalizado)
        datos_normalizados = [(column, valor) for column, valor in data_dict.items()]

        

        # Renderizar la plantilla y pasar los datos al contexto
        return render_template('resultado.html', datos=datos_normalizados, prediccion=prediccion)
    else:
        return render_template('viewAppPrediction.html')"""

""" @app.route('/viewSupportVector',methods=['POST','GET'])
def viewSupportVector():
    if request.method=='POST':
        datos=[request.form.values()]

        json=request.get_json(force=True)
        datos = json['datos']
        
        clasif = joblib.load('modelo.pkl')
        prediccion=clasif.predict(datos)


    return render_template('viewSupportVector.html') 
    if __name__ == '__main__':
    app.run(port=8000)
    
    """


app.run(host='0.0.0.0', port=8000)

