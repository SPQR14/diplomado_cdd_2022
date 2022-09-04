import sys
import os

sys.path.insert(1, os.getcwd())

import practica1_m3

prueba = practica1_m3.Distancias('1')

def prueba_continuas():
    x1 = [i for i in range(10)]
    x2 = [2*i for i in range(10)]
    x3 = [i for i in range(10, 20, 1)]
    print(f'Vectores:\n{x1}\n{x2}\n{x3}')
    print('Distancia Euclidiana')
    print(prueba.euclidean(x1, x2))
    print(prueba.euclidean(x1, x3))
    print(prueba.euclidean(x2, x3))
    print('Distancia Manhattan')
    print(prueba.manhattan(x1, x2))
    print(prueba.manhattan(x1, x3))
    print(prueba.manhattan(x2, x3))
    print('Distancia Minkowski')
    print(prueba.minkowski(x1, x2))
    print(prueba.minkowski(x1, x3))
    print(prueba.minkowski(x2, x3))

def prueba_binarias():
    x1 = [1 if i%2==0 else 0 for i in range(10)]
    x2 = [1 if i%3==0 else 0 for i in range(0, 30, 3)]
    x3 = [1 if i%4==0 else 0 for i in range(0, 40, 4)]
    x4 = [1 if i%5==0 else 0 for i in range(0, 40, 4)]
    x5 = [1 if i%3==0 else 0 for i in range(10)]
    print(f'Vectores:\n{x1}\n{x2}\n{x3}\n{x4}\n{x5}')
    print('Binaria simétrica')
    print(prueba.simetric_binary(x1, x2))
    print(prueba.simetric_binary(x1, x3))
    print(prueba.simetric_binary(x1, x4))
    print(prueba.simetric_binary(x1, x5))
    print(prueba.simetric_binary(x2, x3))
    print(prueba.simetric_binary(x2, x4))
    print(prueba.simetric_binary(x2, x5))
    print(prueba.simetric_binary(x3, x4))
    print(prueba.simetric_binary(x4, x5))
    print("Binaria asimétrica")
    print(prueba.non_simetric_binary(x1, x2))
    print(prueba.non_simetric_binary(x1, x3))
    print(prueba.non_simetric_binary(x1, x4))
    print(prueba.non_simetric_binary(x1, x5))
    print(prueba.non_simetric_binary(x2, x3))
    print(prueba.non_simetric_binary(x2, x4))
    print(prueba.non_simetric_binary(x2, x5))
    print(prueba.non_simetric_binary(x3, x4))
    print(prueba.non_simetric_binary(x4, x5))

def prueba_nominales():
    x1 = ['panda', 'rojo', 'lentes']
    x2 = ['panda', 'rojo', 'sin lentes']
    x3 = ['panda', 'azul', 'lentes']
    x4 = ['león', 'azul', 'lentes']
    x5 = ['perro', 'azul', 'lentes']
    x6 = ['lobo', 'gris', 'lentes']
    x7 = ['lobo', 'gris', 'sin lentes']
    print(f'Vectores:\n{x1}\n{x2}\n{x3}\n{x4}\n{x5}\n{x6}\n{x7}')
    print(prueba.nominal(x1, x2))
    print(prueba.nominal(x1, x3))
    print(prueba.nominal(x1, x4))
    print(prueba.nominal(x1, x5))
    print(prueba.nominal(x1, x6))
    print(prueba.nominal(x1, x7))
    print(prueba.nominal(x2, x3))
    print(prueba.nominal(x2, x4))
    print(prueba.nominal(x2, x5))
    print(prueba.nominal(x2, x6))
    print(prueba.nominal(x2, x7))
    print(prueba.nominal(x3, x4))
    print(prueba.nominal(x3, x5))
    print(prueba.nominal(x3, x6))
    print(prueba.nominal(x4, x7))
    print(prueba.nominal(x5, x6))
    print(prueba.nominal(x5, x7))
    print(prueba.nominal(x6, x7))

def prueba_ordinales():
    x1 = [i for i in range(5)]
    x2 = [i for i in range(10, 15, 1)]
    x3 = [i for i in range(5, 10, 1)]
    print(f'Vectores:\n{x1}\n{x2}\n{x3}')
    print(prueba.ordinal(x1, x2))
    print(prueba.ordinal(x1, x3))
    print(prueba.ordinal(x2, x3))


prueba_continuas()
prueba_binarias()
prueba_nominales()
prueba_ordinales()