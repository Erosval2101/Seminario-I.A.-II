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
df3 = pd.read_csv("salidas_2n.data", header=None)
df4 = pd.read_csv("salidas_3n.data", header=None)

random.seed()

X = df.iloc[0:10].values
y = df2.iloc[0:10].values
sal_2 = df3.iloc[0:4].values
sal_3 = df4.iloc[0:8].values

'''def onclick(event):
    x_1, y_1 = event.xdata, event.ydata
    if not event.inaxes == ax:
        return
    else:
        if event.button is MouseButton.LEFT:
            ax.plot(x_1, y_1, 'ro')
            x1.append(x_1)
            x2.append(y_1)
            y.append(random.randint(0,1))
            plt.show()

        if event.button is MouseButton.RIGHT:
            ax.plot(x_1, y_1, 'bo')
            x1.append(x_1)
            x2.append(y_1)
            y.append(random.randint(0,1))
            plt.show()'''

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

def limpiar(event):
    global X, y
    ax.cla()
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.grid(True)
    X = []
    y = []
    X = df.iloc[0:10].values
    y = df2.iloc[0:10].values
    limpiarPantalla()
    cargar_Datos()

def cargar_Datos():
    global X
    for i in range(len(X)):
        if(X[i, 1] == 0.1 and X[i, 2] == 3.5
        or X[i, 1] == -4 and X[i, 2] == 3.21
        or X[i, 1] == 1.23 and X[i, 2] == 0.12):
            ax.plot(X[i, 1], X[i, 2],'ro')
        if(X[i, 1] == 1.45 and X[i, 2] == -2.21
        or X[i, 1] == 3.54 and X[i, 2] == 1.21):
            ax.plot(X[i, 1], X[i, 2],'bo')
        if(X[i, 1] == -0.68 and X[i, 2] == 2.89):
            ax.plot(X[i, 1], X[i, 2],'go')

def inicializar_Pesos(pesos):
    global X, y
    rango = X.shape[1]
    pesos = np.zeros([len(y[0]), len(y[0])])

def mostrar_Pesos(pesos):
    print("Pesos: ")
    print(np.around(pesos, 4))

def mostrar_Salidas(salidas):
    global X
    print("Salidas: ")
    print(np.around(salidas, 0))
        

def funcion_Activacion(v):
    fv = 1 / (1 + np.exp(-v))
    return fv

def derivada_fact(fv):
    return fv * (1 - fv)

def predict(salida):
    if(salida >= 0.5):
        return 1
    else:
        return 0

def verificar_Error(E, precision):
    errores = 0
    for e in range(len(E)):
        if(np.abs(E[e]) < precision):
            errores += 1
    
    print("Errores: ", errores)
    if(errores == (len(E))):
        return True
    else:
        return False



def graficar_recta(salidas, pesos, n_neuronas, error, epoca, epocas):
    global X
    x_1 = [-5, 5]
    x_2 = []
    lineas = []
    salidas = np.array(salidas)
    salidas = salidas.T
    if(n_neuronas == 2):
        for i in range(X.shape[0]):
            if(salidas[i, 0] == sal_2[0, 0] and salidas[i, 1] == sal_2[0, 1]):
                ax.plot(X[i, 1], X[i, 2], 'bo')
            if(salidas[i, 0] == sal_2[1, 0] and salidas[i, 1] == sal_2[1, 1]):
                ax.plot(X[i, 1], X[i, 2], 'ro')
            if(salidas[i, 0] == sal_2[2, 0] and salidas[i, 1] == sal_2[2, 1]):
                ax.plot(X[i, 1], X[i, 2], 'go')
            if(salidas[i, 0] == sal_2[3, 0] and salidas[i, 1] == sal_2[3, 1]):
                ax.plot(X[i, 1], X[i, 2], 'yo')
        
        n = 0
        while(n < n_neuronas):
            m = pesos[n, 1] / pesos[n, 2]
            b = pesos[n, 0] / pesos[n, 2]

            print(m)
            print(b)

            for e in range(len(x_1)): # calcula x2
                x_2.append(m*x_1[e] + b)
            
            print(x_2)
            if(n == 0):
                line_1 = ax.plot(x_1, x_2, 'r-') # grafica la recta
                plt.pause(0.3)
                line1 = line_1.pop(0)
                #line.remove()
                x_2 = []
            if(n == 1):
                line_2 = ax.plot(x_1, x_2, 'b-') # grafica la recta
                plt.pause(0.3)
                line2 = line_2.pop(0)
                #line.remove()
                x_2 = []
        
        plt.pause(0.3)
        if(error == False):
            if(epoca < epocas-1):
                line1.remove()
                line2.remove()
    
    if (n_neuronas == 3):

        for i in range(X.shape[0]):
            if(salidas[i, 0] == sal_3[0, 0] and salidas[i, 1] == sal_3[0, 1]
            and salidas[i, 1] == sal_3[0, 2]):
                ax.plot(X[i, 1], X[i, 2], 'bo')
            if(salidas[i, 0] == sal_3[1, 0] and salidas[i, 1] == sal_3[1, 1]
            and salidas[i, 1] == sal_3[1, 2]):
                ax.plot(X[i, 1], X[i, 2], 'ro')
            if(salidas[i, 0] == sal_3[2, 0] and salidas[i, 1] == sal_3[2, 1]
            and salidas[i, 1] == sal_3[2, 2]):
                ax.plot(X[i, 1], X[i, 2], 'go')
            if(salidas[i, 0] == sal_3[3, 0] and salidas[i, 1] == sal_3[3, 1]
            and salidas[i, 1] == sal_3[3, 2]):
                ax.plot(X[i, 1], X[i, 2], 'yo')
            if(salidas[i, 0] == sal_3[4, 0] and salidas[i, 1] == sal_3[4, 1]
            and salidas[i, 1] == sal_3[4, 2]):
                ax.plot(X[i, 1], X[i, 2], 'co')
            if(salidas[i, 0] == sal_3[5, 0] and salidas[i, 1] == sal_3[5, 1]
            and salidas[i, 1] == sal_3[5, 2]):
                ax.plot(X[i, 1], X[i, 2], 'mo')
            if(salidas[i, 0] == sal_3[6, 0] and salidas[i, 1] == sal_3[6, 1]
            and salidas[i, 1] == sal_3[6, 2]):
                ax.plot(X[i, 1], X[i, 2], 'ko')
            if(salidas[i, 0] == sal_3[7, 0] and salidas[i, 1] == sal_3[7, 1]
            and salidas[i, 1] == sal_3[7, 2]):
                ax.plot(X[i, 1], X[i, 2], c="orange")
        n = 0
        while(n < n_neuronas):
            m = pesos[n, 1] / pesos[n, 2]
            b = pesos[n, 0] / pesos[n, 2]

            print(m)
            print(b)

            for e in range(len(x_1)): # calcula x2
                x_2.append(m*x_1[e] + b)
            
            print(x_2)
            if(n == 0):
                line_1 = ax.plot(x_1, x_2, 'r-') # grafica la recta
                plt.pause(0.3)
                line1 = line_1.pop(0)
                #line.remove()
                x_2 = []
            if(n == 1):
                line_2 = ax.plot(x_1, x_2, 'b-') # grafica la recta
                plt.pause(0.3)
                line2 = line_2.pop(0)
                #line.remove()
                x_2 = []
            if(n == 2):
                line_3 = ax.plot(x_1, x_2, 'g-') # grafica la recta
                plt.pause(0.3)
                line3 = line_3.pop(0)
                #line.remove()
                x_2 = []
            
            n = n + 1
        
        plt.pause(0.3)
        if(error == False):
            if(epoca < epocas-1):
                line1.remove()
                line2.remove()
                line3.remove()
        



def entrenar_perceptron(event):
    n_entradas = X.shape[0] # tomar entradas
    print("No. de entradas: ", n_entradas)
    n_neuronas = len(y[0]) # tomar número de neuronas
    rango = X.shape[1]
    eta = random.random()
    pesos = np.zeros([len(y[0]), len(y[0])])
    #inicializar_Pesos(pesos)
    
    mostrar_Pesos(pesos)
    
    print("Umbral:")
    for u in range(pesos.shape[0]):
        print("\t {:.2}".format(float(pesos[u, 0])))

    epocas = int(txtEpocas.text)
    epoca = 0
    eta = float(txtEta.text)
    precision = float(txtPrecision.text)
    print("Eta: {:.2}".format(float(eta)))

    E = np.ones([len(y[0])]) # Error de salida
    E_ac = 0 # error actual
    Error_prev = 0 # error anterior
    Ew = np.zeros(len(y[0])) # Error cuadratico medio
    E_red = [] # error de la red
    E_total = 0 # error total
    fv = 0
    epoca = 0
    salida = []
    salidas = []
    Error_minimo = False
    
    while (Error_minimo == False):
        if(epoca == epocas):
            break
        
        print("Epoca actual: ", epoca+1)
        
        n = 0
        while(n < n_neuronas):
            Error_prev = Ew[n]
            for i in range(n_entradas):
                #v = np.dot(X[i,:], pesos.T) + theta
                print("X: ", X[i,1:])
                v = np.dot(X[i,1:], pesos[n,1:]) + pesos[n, 0] # calculo de la salida de la red
                #v += theta

                fv = funcion_Activacion(v)
                sal = predict(fv)
                salida.append(sal)
                E_ac = (y[i, n] - fv) # calculo del error
                print("E ac: ", E_ac)
                dfv =  derivada_fact(fv)
                p = 1
                print(X.shape[1])
                for p in range(X.shape[1]-1):
                    print(p+1)
                    pesos[n,p+1] += (eta * E_ac * X[i,p] * dfv) # ajustar los pesos
               
                pesos[n, 0] += (-(eta * E_ac)) # ajuste de theta

                mostrar_Pesos(pesos)
                print("Umbral: ", pesos[n, 0])
                print("Eta: {:.2}".format(float(eta)))

                E_total = E_total + ((E_ac)**2)
                print("Error total: ", E_total)
            
            Ew[n] = ((1/n_entradas) * (E_total))
            E[n] = (Ew[n] - Error_prev) # Error de la red
            E_red.append(np.abs(E[n]))
            E_total = 0
            salidas.append(salida)
            n = n + 1
            salida = []

        if(verificar_Error(E, precision) == True):
            Error_minimo = True
            graficar_recta(salidas, pesos, n_neuronas, Error_minimo, epoca, epocas)
            print("\n")
            mostrar_Salidas(salidas)
            salidas = []
            plt.pause(0.3)
        if(verificar_Error(E, precision) == False):
            graficar_recta(salidas, pesos, n_neuronas, Error_minimo, epoca, epocas)
            print("\n")
            mostrar_Salidas(salidas)
            salidas = []
            plt.pause(0.3)

        epoca += 1

    plt.pause(0.3)
    print("Epocas completadas")


txtEpocas = TextBox(plt.axes([0.13, 0.05, 0.1, 0.075]), "Epocas")
txtEta = TextBox(plt.axes([0.28, 0.05, 0.1, 0.075]), "Eta")
txtPrecision = TextBox(plt.axes([0.5, 0.05, 0.1, 0.075]), "Precisión")
btnLimpiar = Button( plt.axes([0.65, 0.03, 0.1, 0.1]), 'Limpiar')
btnEntrenar = Button( plt.axes([0.8, 0.03, 0.1, 0.1]), 'Entrenar')

btnEntrenar.on_clicked(entrenar_perceptron)
btnLimpiar.on_clicked(limpiar)
cargar_Datos()
plt.show()
