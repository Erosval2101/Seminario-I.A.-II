import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
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
        pesos.append(random.random())
    
def mostrar_Pesos(pesos):
    for i in range(len(pesos)):
        print("Pesos: {:.2}".format(float(pesos[i])))

def mostrar_Salidas(pesos, n_entradas, theta):
    global X
    salidas = []
    v = 0
    print("Salidas: ")
    for i in range(n_entradas):
        for j in range(X.shape[1]):
            v += ((pesos[j]*X[i][j]))
                
        v = v + theta
        print("{:.4}".format(float(v)))
        


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

    E = 1 # Error de salida
    E_ac = 0 # Error actual
    Error_prev = 0 # Error previo
    Ew = 0 # Error Cuadrático Medio
    E_red = [] # Error en el Adaline
    E_total = 0 # Error total

    for epoca in range(epocas):
        Error_prev = Ew
        acum = -1
        print("\nEpoca: ", epoca+1)
        v = 0
        for i in range(n_entradas):
            for j in range(X.shape[1]):
                v += ((pesos[j]*X[i][j]))
                
            v = v + theta # calculo de la salida
            
            if v >= 0:
                v = 1
            else:
                v = 0


            if v != y[i]:
                E_ac = (y[i] - v)
                for k in range(len(pesos)):
                    pesos[k] = pesos[k] + (X[i][k] * E_ac * eta) # ajuste de pesos
                theta = theta + (-(eta * E_ac)) # ajuste de theta
                E_total = E_total + ((E_ac)**2)

                mostrar_Pesos(pesos)
                print("Eta: {:.2}".format(float(eta)))
                print("Umbral: {:.2}".format(float(theta)))
            else:
                mostrar_Pesos(pesos)
                print("Eta: {:.2}".format(float(eta)))
                print("Umbral: {:.2}".format(float(theta)))
        
        Ew = ((1/n_entradas) * E_total) # Calcular el error cuadratico medio
        E = (Ew - Error_prev) # error del Adaline
        E_red.append(abs(E))

        ax.plot(epoca+1, abs(E), 'bo--')
        print("\nErrores: ", abs(E))
        plt.pause(0.3)


        if((abs(E)) < precision):
            print("\nConvergió en la época: ", epoca+1)
            print("Epocas completadas")
            ax.plot(range(1, len(E_red)+1), E_red, 'bo--', label="Error cuadrático")
            ax.legend(loc='upper right')
            plt.pause(0.3)
            mostrar_Salidas(pesos, n_entradas, theta)
                
            return
    
    mostrar_Salidas(pesos, n_entradas, theta)
    
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