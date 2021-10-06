import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from matplotlib.widgets import TextBox, Button, Cursor
from matplotlib.backend_bases import MouseButton
import os

limpiarPantalla = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.grid(True)

df = pd.read_csv("entrada.data", header=None)
df2 = pd.read_csv("salida.data", header=None)

X = df.iloc[0:10].values
y = df2.iloc[0:10].values
random.seed()

ax.set_ylim(0,2)

def limpiar(event):
    global X, y
    ax.cla()
    ax.set_ylim(0,2)
    ax.grid(True)
    X = []
    y = []
    X = df.iloc[0:10].values
    y = df2.iloc[0:10].values
    limpiarPantalla()

def inicializar_Pesos(pesos):
    global X
    rango = X.shape[1]
    for i in range(rango):
        pesos.append(random.uniform(0, 5))
    
def mostrar_Pesos(pesos):
    for i in range(len(pesos)):
        print("Pesos: {:.4}".format(float(pesos[i])))

def mostrar_Salidas(salidas, n_entradas):
    global X
    print("Salidas: ")
    for i in range(len(salidas)):
        print("{:.4}".format(float(salidas[i])))
        

def funcion_Activacion(v):
    return 1/(1 + np.exp(-v))

def derivada_fact(fv):
    return fv * (1 - fv)

def entrenar_perceptron(event):
    n_entradas = len(y) # tomar entradas
    eta = random.random()
    theta = random.random() # valor de theta
    pesos = []
    inicializar_Pesos(pesos)

    ax.set_ylim(0,2)
    
    mostrar_Pesos(pesos)
    
    print("Umbral: {:.2}".format(float(theta)))

    epocas = int(txtEpocas.text)
    eta = float(txtEta.text)
    precision = float(txtPrecision.text)
    ax.set_xlim(0, epocas+1)
    print("Eta: {:.2}".format(float(eta)))

    E_ac = 0 # Error actual
    E_red = [] # Error en el Adaline
    salidas = []

    pat = 0 # patron
    for epoca in range(epocas):
        error = 0
        print("\nEpoca: ", epoca+1)
        v = 0
        for i in range(n_entradas): # Obtiene salidas
            for j in range(X.shape[1]):
                v += ((pesos[j]*X[i][j]))

            v += theta
            fv = funcion_Activacion(v) # Función de activación
            salidas.append(fv)

            error += math.pow(y[i] - fv, 2) # calcula el error del Adaline

        error = error/n_entradas # calcula el error cuadrático medio
        E_red.append(abs(error))
        ax.plot(epoca+1, abs(error), 'bo--') # grafica el error cuadrático medio de la época
        print("\nError: {:.4}".format(float(abs(error))))
        plt.pause(0.3)

        if ((abs(error)) < precision): # Condición de paro, si el error cuadrático medio es menor a la precisión
            print("\nConvergió en la época: ", epoca+1)
            print("Epocas completadas")
            ax.plot(range(1, len(E_red)+1), E_red, 'bo--', label="Error cuadrático")
            ax.legend(loc='upper right')
            plt.pause(0.3)
            mostrar_Salidas(salidas, n_entradas)
            return
        else:
            v = 0
            for j in range(X.shape[1]):
                v += ((pesos[j]*X[pat][j]))

            v += theta
            fv = funcion_Activacion(v)


            dfv = derivada_fact(fv) # Se obtiene la derivada de la función de activación
            E_ac = (y[pat] - fv) # Error actual
            for k in range(len(pesos)):
                pesos[k] = pesos[k] + (X[pat][k] * E_ac * eta * dfv) # ajuste de pesos
            
            theta = theta + (-(eta * E_ac)) # ajuste de theta
            pat = pat + 1

            mostrar_Pesos(pesos)
            print("Eta: {:.2}".format(float(eta)))
            print("Umbral: {:.2}".format(float(theta)))
        
        if pat >= n_entradas:
            pat = 0
        
        mostrar_Salidas(salidas, n_entradas)
        salidas = []
    
    ax.plot(range(1, len(E_red)+1), E_red, 'bo--', label="Error cuadrático")
    ax.legend(loc='upper right')
    plt.pause(0.3)
    print("\nNo se pudo converger")
    print("Epocas completadas")


errores = []
txtEpocas = TextBox(plt.axes([0.13, 0.05, 0.1, 0.075]), "Epocas")
txtEta = TextBox(plt.axes([0.28, 0.05, 0.1, 0.075]), "Eta")
txtPrecision = TextBox(plt.axes([0.5, 0.05, 0.1, 0.075]), "Precisión")
btnLimpiar = Button( plt.axes([0.65, 0.03, 0.1, 0.1]), 'Limpiar')
btnEntrenar = Button( plt.axes([0.8, 0.03, 0.1, 0.1]), 'Entrenar')

btnEntrenar.on_clicked(entrenar_perceptron)
btnLimpiar.on_clicked(limpiar)
plt.show()