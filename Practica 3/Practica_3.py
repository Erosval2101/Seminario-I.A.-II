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

ax.set_ylim(0,10)

def limpiar(event):
    global X, y
    ax.cla()
    ax.set_ylim(0,10)
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


def entrenar_perceptron(event):
    n_entradas = len(y) # tomar entradas
    eta = random.random()
    theta = random.random() # valor de theta
    pesos = []
    inicializar_Pesos(pesos)
    salidas = []

    ax.set_ylim(0,10)
    
    mostrar_Pesos(pesos)
    
    print("Umbral: {:.2}".format(float(theta)))

    epocas = int(txtEpocas.text)
    eta = float(txtEta.text)
    ax.set_xlim(0, epocas+1)
    print("Eta: {:.2}".format(float(eta)))
    for epoca in range(epocas):
        acum = -1
        print("\nEpoca: ", epoca+1)
        v = 0
        for i in range(n_entradas):
            for j in range(X.shape[1]):
                v += ((pesos[j]*X[i][j]))
            
            v = v - theta
            if v >= 0:
                v = 1
                salidas.append(v)
            else:
                v = 0
                salidas.append(v)

            if v != y[i]:
                error = (y[i] - v)
                acum += abs(error)
                theta = theta + (-(eta * error))
                for k in range(len(pesos)):
                    pesos[k] = pesos[k] + (X[i][k] * error * eta)
                
                mostrar_Pesos(pesos)
                print("Eta: {:.2}".format(float(eta)))
                print("Umbral: {:.2}".format(float(theta)))
            else:
                mostrar_Pesos(pesos)
                print("Eta: {:.2}".format(float(eta)))
                print("Umbral: {:.2}".format(float(theta)))
            

        errores.append(acum)
        ax.plot(epoca+1, acum, 'bo--')
        print("\nErrores: ", acum)
        plt.pause(0.3)

        for i in range(len(salidas)):
            print("Salidas: ",salidas[i])

        if(acum == 0):
            print("\nConvergió en la época: ", epoca+1)
            print("Epocas completadas")
            ax.plot(range(1, len(errores)+1), errores, 'bo--')
            plt.pause(0.3)
            for i in range(len(salidas)):
                print("Salidas: ",salidas[i])
            
            salidas = []
            return
        
        salidas = []
    for i in range(len(salidas)):
        print("Salidas: ",salidas[i])
    
    ax.plot(range(1, len(errores)+1), errores, 'bo--')
    plt.pause(0.3)
    print("\nNo se pudo converger")
    print("Epocas completadas")


errores = []
txtEpocas = TextBox(plt.axes([0.15, 0.05, 0.1, 0.075]), "Epocas")
txtEta = TextBox(plt.axes([0.35, 0.05, 0.1, 0.075]), "Eta")
btnLimpiar = Button( plt.axes([0.55, 0.03, 0.1, 0.1]), 'Limpiar')
btnEntrenar = Button( plt.axes([0.8, 0.03, 0.1, 0.1]), 'Entrenar')

btnEntrenar.on_clicked(entrenar_perceptron)
btnLimpiar.on_clicked(limpiar)
plt.show()
