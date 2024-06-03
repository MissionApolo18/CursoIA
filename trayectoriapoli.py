#Sección 1
import numpy as np
import matplotlib.pyplot as plt
#Sección 2
l1 = l2 = 0.15  # Longitudes de los eslabones en metros
xi, yi = 0.15, 0.15  # Posición inicial en metros
ti, tf = 0, 15  # Tiempo inicial y final
T = 0.002  # Periodo de muestreo
#Sección 3
def get_user_input():
    try:
        xf = float(input("Ingrese la posición deseada en x (en metros, menor a 0.15): "))
        yf = float(input("Ingrese la posición deseada en y (en metros, menor a 0.15): "))
        if np.sqrt(xf**2 + yf**2) > (l1 + l2):
            raise ValueError("La posición final está fuera del espacio de trabajo del robot.")
        return xf, yf
    except ValueError as e:
        print(f"Entrada no válida: {e}")
        exit()
#Sección 4
def calcular_coeficientes(xi, yi, xf, yf, ti, tf):
    A = np.array([
        [1, ti, ti**2, ti**3, ti**4, ti**5],
        [0, 1, 2*ti, 3*ti**2, 4*ti**3, 5*ti**4],
        [0, 0, 2, 6*ti, 12*ti**2, 20*ti**3],
        [1, tf, tf**2, tf**3, tf**4, tf**5],
        [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
        [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
    ])
    bx = np.array([xi, 0, 0, xf, 0, 0])
    by = np.array([yi, 0, 0, yf, 0, 0])
    a_x = np.linalg.solve(A, bx)
    a_y = np.linalg.solve(A, by)
    return a_x, a_y
#Seccion 5
def generar_trayectoria(a_x, a_y, t):
    xd = a_x[0] + a_x[1]*t + a_x[2]*t**2 + a_x[3]*t**3 + a_x[4]*t**4 + a_x[5]*t**5
    yd = a_y[0] + a_y[1]*t + a_y[2]*t**2 + a_y[3]*t**3 + a_y[4]*t**4 + a_y[5]*t**5
    return xd, yd
#Sección 6
def plot_trajectoria(t, xd, yd):
    plt.figure(figsize=(12, 6))
    #
    plt.subplot(1, 2, 1)
    plt.plot(t, xd, label='Trayectoria en x')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Posición en x [m]')
    plt.title('Trayectoria Polinomial en x')
    plt.legend()
    plt.grid(True)
    #
    plt.subplot(1, 2, 2)
    plt.plot(t, yd, label='Trayectoria en y')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Posición en y [m]')
    plt.title('Trayectoria Polinomial en y')
    plt.legend()
    plt.grid(True)
    #
    plt.tight_layout()
    plt.show()
#Sección 7
def main():
    xf, yf = get_user_input()
    print("Posición deseada en x:", xf)
    print("Posición deseada en y:", yf)
    t = np.arange(ti, tf, T)
    a_x, a_y = calcular_coeficientes(xi, yi, xf, yf, ti, tf)
    xd, yd = generar_trayectoria(a_x, a_y, t)
    plot_trajectoria(t, xd, yd)
#sección 8
main()