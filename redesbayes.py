# Definir probabilidades
P_homicida = 0.01
P_no_homicida = 1 - P_homicida

P_sangre_given_homicida = 0.8
P_no_sangre_given_homicida = 1 - P_sangre_given_homicida

P_sangre_given_no_homicida = 0.1
P_no_sangre_given_no_homicida = 1 - P_sangre_given_no_homicida

P_cuchillo_given_homicida = 0.85
P_no_cuchillo_given_homicida = 1 - P_cuchillo_given_homicida

P_cuchillo_given_no_homicida = 0.25
P_no_cuchillo_given_no_homicida = 1 - P_cuchillo_given_no_homicida

# Funciones para realizar consultas
def P_sangre():
    return P_sangre_given_homicida * P_homicida + P_sangre_given_no_homicida * P_no_homicida

def P_cuchillo():
    return P_cuchillo_given_homicida * P_homicida + P_cuchillo_given_no_homicida * P_no_homicida

def P_no_cuchillo():
    return 1 - P_cuchillo()

def P_no_cuchillo_given_homicida():
    return 1 - P_cuchillo_given_homicida

def P_homicida_given_cuchillo_y_sangre():
    P_cuchillo_y_sangre_given_homicida = P_cuchillo_given_homicida * P_sangre_given_homicida
    P_cuchillo_y_sangre_given_no_homicida = P_cuchillo_given_no_homicida * P_sangre_given_no_homicida
    P_cuchillo_y_sangre = P_cuchillo_y_sangre_given_homicida * P_homicida + P_cuchillo_y_sangre_given_no_homicida * P_no_homicida
    return (P_cuchillo_y_sangre_given_homicida * P_homicida) / P_cuchillo_y_sangre

# Realizar consultas
def tarea():
    print("P(homicida) = " + str(P_homicida))
    print("P(sangre) = " + str(P_sangre()))
    print("P(cuchillo) = " + str(P_cuchillo()))
    print("P(~cuchillo) = " + str(P_no_cuchillo()))
    print("P(~cuchillo|homicida) = " + str(P_no_cuchillo_given_homicida()))
    print("P(homicida|cuchillo,sangre) = " + str(P_homicida_given_cuchillo_y_sangre()))

tarea()
