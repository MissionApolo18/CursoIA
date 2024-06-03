#Sección 1
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
#Sección 2
l1 = l2 = 0.15  # Longitudes de los eslabones en metros
xi, yi = 0.15, 0.15  # Posición inicial en metros
ti, tf = 0, 15  # Tiempo inicial y final
T = 0.002  # Periodo de muestreo
#Sección 3
def get_user_input():
    try:
        root = tk.Tk()
        root.title("Ingrese las posiciones deseadas")

        label_xf = tk.Label(root, text="Posición final en x (menor a 0.15):")
        label_xf.pack()
        entry_xf = tk.Entry(root)
        entry_xf.pack()

        label_yf = tk.Label(root, text="Posición final en y (menor a 0.15):")
        label_yf.pack()
        entry_yf = tk.Entry(root)
        entry_yf.pack()

        def submit():
            try:
                xf = float(entry_xf.get())
                yf = float(entry_yf.get())
                if xf < 0 or yf < 0 or np.sqrt(xf**2 + yf**2) > (l1 + l2):
                    raise ValueError("La posición final está fuera del espacio de trabajo del robot.")
                global final_positions
                final_positions = (xf, yf)
                print("Valores finales ingresados:", final_positions)  # Agregar mensaje de impresión
                root.quit()  # Salir del bucle principal de la ventana
            except ValueError as e:
                messagebox.showerror("Error", f"Entrada no válida: {e}")

        button_submit = tk.Button(root, text="Aceptar", command=submit)
        button_submit.pack()

        root.mainloop()  # Iniciar el bucle principal de la ventana
        root.destroy()  # Destruir la ventana después de salir del bucle principal

        return final_positions

    except Exception as e:
        print(f"Ocurrió un error: {e}")
        return None
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
def plot_trajectoria(t, xd, yd, q1d, q2d, tau_1, tau_2):
    """Graficar las trayectorias polinomiales en x, y y las trayectorias articulares."""
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 2, 1)
    plt.plot(t, xd, label='Trayectoria en x')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Posición en x [m]')
    plt.title('Trayectoria Polinomial en x')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(t, yd, label='Trayectoria en y')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Posición en y [m]')
    plt.title('Trayectoria Polinomial en y')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(t, q1d, label='q1d (ángulo)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Ángulo q1d [grados]')
    plt.title('Trayectoria Articular q1d')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(t, q2d, label='q2d (ángulo)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Ángulo q2d [grados]')
    plt.title('Trayectoria Articular q2d')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(t, tau_1, label='τ1 (Control)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('τ1')
    plt.title('Señal de Control τ1')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(t, tau_2, label='τ2 (Control)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('τ2')
    plt.title('Señal de Control τ2')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
#Sección 7
def cinematica_inversa(xd, yd, l1, l2):
    cos_teta2 = (xd**2 + yd**2 - l1**2 - l2**2) / (2 * l1 * l2)
    sin_teta2 = np.sqrt(1 - cos_teta2**2)  

    teta2 = np.arctan2(sin_teta2, cos_teta2)
    #Teta1
    k1 = l1 + l2 * cos_teta2
    k2 = l2 * sin_teta2
    teta1 = np.arctan2(yd, xd) - np.arctan2(k2, k1)

    return np.degrees(teta1), np.degrees(teta2)
#Sección 8
def pd_controller(qtil1, qtil2, t):
    """Implementar el controlador PD para q_tilde_1 y q_tilde_2."""
    # Ganancias del controlador (unitarias para proporcional)
    K_p1 = K_p2 = 1
    K_D1 = K_D2 = 1

    dq_tilde_1_dt = np.cos(t)
    dq_tilde_2_dt = -np.sin(t)

    tau_1 = K_p1 * qtil1 + K_D1 * dq_tilde_1_dt
    tau_2 = K_p2 * qtil2 + K_D2 * dq_tilde_2_dt

    return tau_1, tau_2
#Sección 9
def main():
    xf, yf = get_user_input()
    if xf is None or yf is None:
        return
    t = np.arange(ti, tf, T)
    a_x, a_y = calcular_coeficientes(xi, yi, xf, yf, ti, tf)
    xd, yd = generar_trayectoria(a_x, a_y, t)
    # Convertir xd y yd de metros a centímetros
    xd_cm = xd * 100
    yd_cm = yd * 100
    #Pruebaaaa
    q1d, q2d = cinematica_inversa(xd, yd, l1, l2)
    
    # Señales de error
    q_tilde_1 = np.sin(t)
    q_tilde_2 = np.cos(t)
    
    # Control PD
    tau_1, tau_2 = pd_controller(q_tilde_1, q_tilde_2, t)
    
    plot_trajectoria(t, xd_cm, yd_cm, q1d, q2d,tau_1,tau_2)
#sección 10
plt.ioff()
main()