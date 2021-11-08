import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib.backend_bases import MouseButton
import os
import csv

limpiarPantalla = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.grid(True)

X = np.zeros((2, 0))
Y = np.zeros(0)


def sigmoide(v, derivada=False):
    a = 1/(1+ np.exp(-v))
    if derivada:
        da = np.ones(v.shape)
        return a, da
    return a

def tanh(v, derivada=False):
    a = np.tanh(v)
    if derivada:
        da = (1 - a) * (1 + a)
        return a, da
    return a


def sigmoide_oculta(v, derivada=False):
    a  = 1/(1 + np.exp(-v))
    if derivada:
        da = a * (1 - a)
        return a, da
    return a


class MLP:

    def __init__(self,capas_dim,act_ocultas=tanh,act_salida=sigmoide):
        self.C = len(capas_dim) - 1
        self.pesos = [None] * (self.C + 1)
        self.b = [None] * (self.C + 1)
        self.f = [None] * (self.C + 1)

        for c in range(1, self.C+1):
            self.pesos[c] = -1 + 2 * np.random.rand(capas_dim[c],capas_dim[c-1])
            self.b[c] = -1 + 2 * np.random.rand(capas_dim[c],1)
            if c == self.C:
                self.f[c] = act_salida
            else:
                self.f[c] = act_ocultas

    def entrenar(self, X, Y, epocas, eta):
        P = X.shape[1]

        for e in range(epocas):
            print("Epoca actual: ", e+1)
            for p in range(P):
                a = [None] * (self.C + 1)
                da = [None] * (self.C + 1)
                grad_local = [None] * (self.C + 1)

                a[0] = X[:,p].reshape(-1,1)
                for c in range(1,self.C + 1):
                    v = np.dot(self.pesos[c],a[c-1]) + self.b[c]
                    a[c] , da[c] = self.f[c](v, derivada=True)

                for c in range(self.C,0,-1):
                    if c == self.C:
                        grad_local[c] = (Y[:,p].reshape(-1,1) - a[c]) * da[c]
                    else:
                        grad_local[c] = np.dot(self.pesos[c+1].T, grad_local[c+1]) * da[c]

                for c in range(1,self.C+1):
                    self.pesos[c] += eta * np.dot(grad_local[c],a[c-1].T)
                    self.b[c] += eta * grad_local[c]
            
            # Actualizar gráfica
            xmin, ymin = np.min(X[0,:])-0.2, np.min(X[1,:])-0.2
            xmax, ymax = np.max(X[0,:])+0.2, np.max(X[1,:])+0.2
            xx, yy = np.meshgrid(np.linspace(xmin,xmax,100),
                                np.linspace(ymin,ymax,100))
            data = [xx.ravel(),yy.ravel()]
            zz = net.predict(data)
            zz = zz.reshape(xx.shape)
            ax.contourf(xx,yy,zz,alpha=0.8,
                    cmap=matplotlib.cm.get_cmap("RdBu"))
            
            plt.pause(0.0000000001)
            fig.canvas.blit(fig.bbox)
            fig.canvas.resize_event()
    
    def predict(self,X):
        a = np.asanyarray(X)
        for c in range(1,self.C+1):
            v = np.dot(self.pesos[c],a) + self.b[c]
            a = self.f[c](v)
        return a

def limpiar(event):
    global X, Y
    ax.cla()
    limpiarPantalla()
    X = np.zeros((2, 0))
    Y = np.zeros(0)
    n_Archivo = str(input("Ingrese el archivo con el conjunto de datos: "))
    numCapas = int(input("Ingrese el número de capas: "))
    capas = [None] * (numCapas + 1)
    for i in range(numCapas + 1):
        if(i == 0):
            capas[i] = 2
        elif(i == numCapas):
            neuronas = int(input("Ingrese el número de neuronas de la última capa: "))
            capas[i] = neuronas
        else:
            neuronas = int(input("Ingrese el número de neuronas de la capa No. " + str(i)+ ": " ))
            print("\n")
            capas[i] = neuronas

    net = MLP(tuple(capas))
    cargar_datos(n_Archivo)
    cargar_grafica(X,Y,net)

def cargar_datos(nArchivo):
    global X, Y
    with open(nArchivo, 'r') as archivo:
        arch = csv.reader(archivo, delimiter=',')
        for fila in arch:
            X = np.append(X, [[float(fila[0])], [float(fila[1])]], axis=1)
            Y = np.append(Y, [[float(fila[2])]])
    Y = Y.reshape(1,-1)

def cargar_grafica(X,Y,net):

    for i in range(X.shape[1]):
        if(Y[0, i] == 0):
            ax.plot(X[0,i], X[1,i], 'ro')
        else:
            ax.plot(X[0,i], X[1,i], 'bo')

    
    xmin, ymin = np.min(X[0,:])-0.2, np.min(X[1,:])-0.2
    xmax, ymax = np.max(X[0,:])+0.2, np.max(X[1,:])+0.2
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,100),
                         np.linspace(ymin,ymax,100))
    
    data = [xx.ravel(),yy.ravel()]
    zz = net.predict(data)
    zz = zz.reshape(xx.shape)
    
    ax.contourf(xx,yy,zz,alpha=0.8,
                cmap=matplotlib.cm.get_cmap("RdBu"))
    
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.grid()

def entrenar_perceptron(event):
    epocas = int(txtEpocas.text)
    eta = float(txtEta.text)
    net.entrenar(X,Y,epocas,eta)

txtEpocas = TextBox(plt.axes([0.13, 0.05, 0.1, 0.075]), "Epocas")
txtEta = TextBox(plt.axes([0.28, 0.05, 0.1, 0.075]), "Eta")
btnLimpiar = Button( plt.axes([0.65, 0.03, 0.1, 0.1]), 'Limpiar')
btnEntrenar = Button( plt.axes([0.8, 0.03, 0.1, 0.1]), 'Entrenar')

nArchivo = str(input("Ingrese el archivo con el conjunto de datos: "))
numCapas = int(input("Ingrese el número de capas: "))
capas = [None] * (numCapas + 1)
for i in range(numCapas + 1):
    if(i == 0):
        capas[i] = 2
    elif(i == numCapas):
        neuronas = int(input("Ingrese el número de neuronas de la última capa: "))
        capas[i] = neuronas
    else:
        neuronas = int(input("Ingrese el número de neuronas de la capa No. " + str(i)+ ": " ))
        print("\n")
        capas[i] = neuronas

net = MLP(tuple(capas))
cargar_datos(nArchivo)
cargar_grafica(X,Y,net)

btnEntrenar.on_clicked(entrenar_perceptron)
btnLimpiar.on_clicked(limpiar)
plt.show()



