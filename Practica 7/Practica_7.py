import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib.backend_bases import MouseButton

fig, (ax_main, ax_error) = plt.subplots(1, 2)
fig.set_size_inches(10, 6, forward=True)
fig.subplots_adjust(bottom=0.2)
ax_main.grid(True)


x = np.linspace(-3,3,100)
x_size = x.size
y = np.zeros((x_size,1))
for i in range(x_size):
    y[i]= math.sin(x[i])

ax_main.plot(x, y)

def tanh(v, derivada=False):
    a = np.tanh(v)
    if derivada:
        da = (1 - a) * (1 + a)
        return a, da
    return a 


def entrenar_perceptron(event):
    epocas = int(txtEpocas.text)+1
    neuronas = int(txtNeuronas.text)
    W1 = np.random.random ((neuronas, 1)) 
    B1 = np.random.random ((neuronas, 1)) 
    W2 = np.random.random ((1, neuronas))
    B2 = np.random.random ((1,1)) 
    threshold = 0.005

    E = np.zeros((epocas, 1))
    Y = np.zeros((x_size, 1))
    for k in range(epocas):
        temp = 0
        for i in range(x_size):
            hide_in = np.dot(x[i], W1) - B1 # datos de entrada de capa oculta
            hide_out = np.zeros((neuronas, 1)) #datos de salida de capa
            for j in range(neuronas):
                hide_out[j] = tanh(hide_in[j])
                y_out = np.dot(W2, hide_out) -B2 # salida del modelo
    
            Y[i] = y_out

            e = y_out-y[i] # salida del modelo menos el resultado real. Error de dibujo

            dB2 = -1*threshold*e
            dW2 = e*threshold*np.transpose(hide_out)
            dB1 = np.zeros((neuronas,1))
            for j in range(neuronas):
                dB1[j] = np.dot(np.dot(W2[0][j],tanh(hide_in[j])),(1-tanh(hide_in[j]))*(-1)*e*threshold)
    
            dW1 = np.zeros((neuronas,1))
    
            for j in range(neuronas):
                dW1[j] = np.dot(np.dot(W2[0][j],tanh(hide_in[j])),(1-tanh(hide_in[j]))*x[i]*e*threshold)
    
            W1 = W1 - dW1
            B1 = B1 - dB1
            W2 = W2 - dW2
            B2 = B2 - dB2
            temp = temp + abs(e)

        E[k] = temp
        print("Error: ", E[k])
    
        if k%100==0:
            print("Epoca: ", k)
    
    ax_error.plot(range(1, len(E)+1), E, 'bo--')
    ax_main.plot(x, Y)


txtEpocas = TextBox(plt.axes([0.13, 0.05, 0.1, 0.075]), "Epocas")
txtNeuronas = TextBox(plt.axes([0.38, 0.05, 0.1, 0.075]), "No. Neuronas")
btnEntrenar = Button( plt.axes([0.8, 0.03, 0.1, 0.1]), 'Entrenar')

btnEntrenar.on_clicked(entrenar_perceptron)

plt.show()
