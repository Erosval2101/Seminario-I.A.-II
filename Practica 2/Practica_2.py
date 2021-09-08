import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib.widgets import TextBox, Button

data = pd.read_csv('datos.csv', header=None) # leer archivo
x1 = data.iloc[:, 0]
x2 = data.iloc[:, 1]
y = data.iloc[:, -1]

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

n_entradas = len(y) # tomar entradas

theta = random.random() # valor de theta
w1 = random.random() # pesos
w2 = random.random()
m = -w1/w2
b = theta/w2
x_1 = [min(x1), max(x1)] # toma valor máximo y mínimo de x1 para graficar la recta
x_2 = []
for i in range(n_entradas):
    v = ((w1*x1[i]) + (w2*x2[i])) - theta # calcular v
    if v >= 0: # si v >= 0 pone un punto azul
        plt.plot(x1[i], x2[i], "bs")
        plt.draw()
        plt.pause(0.2)
    else: # sino, pone un punto rojo
        plt.plot(x1[i], x2[i], "r^")
        plt.draw()
        plt.pause(0.2)

for e in range(len(x_1)): # calcula x2
    x_2.append(m*x_1[e] + b)
            
plt.plot(x_1, x_2, 'y-') # grafica la recta
ax.set_title('Perceptrón')
axTxtEpocas = fig.add_axes([0.15, 0.05, 0.1, 0.075])
axTxtEta = fig.add_axes([0.35, 0.05, 0.1, 0.075])
axBtnEntrenar = fig.add_axes([0.8, 0.03, 0.1, 0.1])
print("Pesos: {0:.2f}".format(w1)+ " {0:.2f}".format(w2))
print("Umbral: {0:.2f}".format(theta))
txtEpocas = TextBox(axTxtEpocas, "Epocas")
txtEta = TextBox(axTxtEta, "Eta")
btnEntrenar = Button( axBtnEntrenar, 'Entrenar')
plt.xlim(-8,8)
plt.ylim(-8,8)


def entrenar_perceptron(event):
    plt.clf()
    global w1, w2, b, theta, x_1, m
    epocas = int(txtEpocas.text)
    eta = float(txtEta.text)
    print("Eta: {0:.2f}".format(eta))
    for epoca in range(epocas):
        errores = 0
        print("\nEpoca: ", epoca+1)
        for i in range(n_entradas):
            x_2 = []
            v = ((w1*x1[i]) + (w2*x2[i])) - theta # calcular v
            if v >= 0:
                plt.plot(x1[i], x2[i], "bs")
                plt.draw()
                v = 1
            else:
                plt.plot(x1[i], x2[i], "r^")
                plt.draw()
                v = 0

            if v != y[i]:
                error = (y[i] - v)
                theta = theta + (-(eta * error))
                w1 = w1 + (x1[i] * error * eta)
                w2 = w2 + (x2[i] * error * eta)
                errores+=1
                b = theta/w2
                m = -w1/w2
                print("Pesos: {0:.2f}".format(w1)+ " {0:.2f}".format(w2))
                print("Eta: {0:.2f}".format(eta))
                print("Umbral: {0:.2f}".format(theta))

            for e in range(len(x_1)): # calcula x2
                x_2.append(m*x_1[e] + b)            

            line_1 = plt.plot(x_1, x_2, 'y-') # grafica la recta
            plt.pause(0.3)
            line = line_1.pop(0)
            line.remove()

    print("Epocas completadas")

btnEntrenar.on_clicked(entrenar_perceptron)

plt.show()