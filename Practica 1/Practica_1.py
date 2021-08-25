import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def opcion_or():
    data = pd.read_csv('datos_or.csv', header=None) # leer archivo
    x1 = data.iloc[:, 0]
    x2 = data.iloc[:, 1]
    y = data.iloc[:, -1]

    n_entradas = len(y) # tomar entradas

    theta = 0.5 # valor de theta en compuerta OR
    w1 = w2 = 1 # pesos
    m = -w1/w2
    b = theta/w2
    x_1 = [min(x1), max(x1)] # toma valor máximo y mínimo de x1 para graficar la recta
    x_2 = []
    for i in range(n_entradas):
        v = ((w1*x1[i]) + (w2*x2[i])) - theta # calcular v
        if v >= 0: # si v >= 0 pone un punto azul
            plt.plot(x1[i], x2[i], "bs")
        else: # sino, pone un punto rojo
            plt.plot(x1[i], x2[i], "r^")

    for e in range(len(x_1)): # calcula x2
        x_2.append(m*x_1[e] + b)
        print(x_2[e])
            
    plt.plot(x_1, x_2, 'y-') # grafica la recta
    plt.show()


def opcion_and():
    data = pd.read_csv('datos_and.csv', header=None) # leer archivo
    x1 = data.iloc[:, 0]
    x2 = data.iloc[:, 1]
    y = data.iloc[:, -1]

    n_entradas = len(y) # tomar entradas

    theta = 1.5 # valor de theta usando compuerta AND
    w1 = w2 = 1 # pesos
    m = -w1/w2
    b = theta/w2
    x_1 = [min(x1), max(x1)] # toma valor máximo y mínimo de x1 para graficar la recta
    x_2 = []
    for i in range(n_entradas):
        v = ((w1*x1[i]) + (w2*x2[i])) - theta #calcular v
        if v >= 0: # si v >= 0 pone un punto azul
            plt.plot(x1[i], x2[i], "bs")
        else: # sino, pone un punto rojo
            plt.plot(x1[i], x2[i], "r^")

    for e in range(len(x_1)): # calcula x2
        x_2.append(m*x_1[e] + b)
        print(x_2[e])
            
    plt.plot(x_1, x_2, 'y-') # grafica la recta
    plt.show()

def main():
    opc = 0
    while opc != 3:
        print("=== Práctica 1: Neurona AND y OR ===")
        print("Seminario de Solución de Problemas de Inteligencia Artificial II")
        print("=== Menú ===")
        print("1-. AND")
        print("2-. OR")
        print("3-. Salir")
        opc = int(input("Seleccione una opcion: "))
        if opc == 1:
            opcion_and()
        if opc == 2:
            opcion_or()
        if opc == 3:
            exit()

if __name__ == "__main__":
    main()

