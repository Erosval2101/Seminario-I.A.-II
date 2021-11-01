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

random.seed()

x1 = []
x2 = []
y = []

cursor = Cursor(ax,
                horizOn=False,
                vertOn=False)

def onclick(event):
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
            plt.show()

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

def limpiar(event):
    global x1, x2, y
    ax.cla()
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    x1 = []
    x2 = []
    y = []
    ax.grid(True)
    limpiarPantalla()


def entrenar_perceptron(event):
    n_entradas = len(y) # tomar entradas

    theta = random.random() # valor de theta
    w1 = random.random() # pesos
    w2 = random.random()
    m = -w1/w2
    b = theta/w2
    x_1 = [-5, 5] # toma valor máximo y mínimo de x1 para graficar la recta
    x_2 = []

    print("Pesos: {0:.2f}".format(w1)+ " {0:.2f}".format(w2))
    print("Umbral: {0:.2f}".format(theta))

    #plt.xlim(-1,1)
    #plt.ylim(-1,1)

    epocas = int(txtEpocas.text)
    eta = float(txtEta.text)
    print("Eta: {0:.2f}".format(eta))
    for epoca in range(epocas):
        acum = -1
        print("\nEpoca: ", epoca+1)
        x_2 = []
        for i in range(n_entradas):
            v = ((w1*x1[i]) + (w2*x2[i])) - theta # calcular v
            if v >= 0:
                ax.plot(x1[i], x2[i], 'bo')
                v = 1
            else:
                ax.plot(x1[i], x2[i], 'ro')
                v = 0

            if v != y[i]:
                error = (y[i] - v)
                acum += abs(error)
                theta = theta + (-(eta * error))
                w1 = w1 + (x1[i] * error * eta)
                w2 = w2 + (x2[i] * error * eta)
                b = theta/w2
                m = -w1/w2
                print("Pesos: {0:.2f}".format(w1)+ " {0:.2f}".format(w2))
                print("Eta: {0:.2f}".format(eta))
                print("Umbral: {0:.2f}".format(theta))
            else:
                print("Pesos: {0:.2f}".format(w1)+ " {0:.2f}".format(w2))
                print("Eta: {0:.2f}".format(eta))
                print("Umbral: {0:.2f}".format(theta))
            

        for e in range(len(x_1)): # calcula x2
            x_2.append(m*x_1[e] + b)

        line_1 = ax.plot(x_1, x_2, 'g-') # grafica la recta
        plt.pause(0.3)
        line = line_1.pop(0)
        errores.append(acum)
        print("\nErrores: ", acum)

        if(acum == 0):
            print("\nConvergió en la época: ", epoca+1)
            print("Epocas completadas")

            for i in range(n_entradas):
                v = ((w1*x1[i]) + (w2*x2[i])) - theta # calcular v
                if v >= 0:
                    ax.plot(x1[i], x2[i], 'bo')
                    v = 1
                else:
                    ax.plot(x1[i], x2[i], 'ro')
                    v = 0

            x_2 = []
            for e in range(len(x_1)): # calcula x2
                x_2.append(m*x_1[e] + b)

            line_1 = ax.plot(x_1, x_2, 'y-') # grafica la recta
            plt.pause(0.3)
            line = line_1.pop(0)
            return

    print("\nNo se pudo converger")
    print("Epocas completadas")


errores = []
txtEpocas = TextBox(plt.axes([0.15, 0.05, 0.1, 0.075]), "Epocas")
txtEta = TextBox(plt.axes([0.35, 0.05, 0.1, 0.075]), "Eta")
btnLimpiar = Button( plt.axes([0.55, 0.03, 0.1, 0.1]), 'Limpiar')
btnEntrenar = Button( plt.axes([0.8, 0.03, 0.1, 0.1]), 'Entrenar')

btnEntrenar.on_clicked(entrenar_perceptron)
btnLimpiar.on_clicked(limpiar)
plt.connect('button_press_event', onclick)
plt.show()
