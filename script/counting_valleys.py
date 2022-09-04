#!/bin/bash
# -*- coding: utf-8 -*-

"""
counting_valleys.py: Diplomado ed Ciencia de datos Módulo 3 - Trabajo extra.

La idea es contar el número de valles definidos en un string. El string tiene la característica de que únicamente trae dos  caracteres diferentes, U ó D. En secuencia consecutiva. 

La letra D, significa descender un nivel. Mientras que la letra U representa subir un nivel. Cada Valle se define como el viaje de descender n niveles y después subir n hasta llegar a la altura original. No import si dentro de la secuencia hay una ligera subida (DDDUUD, aquí bajas 3, subes 2 vuelves a bajar 1) pues no llegó a la altura original entonces no se cuenta como un Valle. 

La idea es que la función reciba un string compuesto de puras U y D, y arroje la cantidad de valles encontrados en este string.

Como regla, no se puede utilizar ninguna librería o paquetería, simplemente su código python puro con sus estructuras y arreglos. 

La función será evaluada por 10 strings, y tiene que poder ejecutar en Google Colab (estándar) en menos de 10 segundos para cada string probado. Considere que el string puede llegar a tener hasta 10,000,000 caracteres. Y mínimo debe de tener 2.
"""

__author__ = "Alberto Isaac Pico Lara"
__date__ = "domingo 4/09/2022"

from itertools import count


def valley_counts(cadena:str):

    # Verificamos la longitud de la cadena, según el requirimiento puede ser de 2 a 10,000,000
    if len(cadena) < 2 or len(cadena) > 10000000:
        return 'Error: Out of accepted limits.'

    # Estandarizar y limpiar la cadena
    cadena = cadena.upper().lstrip().strip()

    nivel = 0
    valles = 0
    for i in cadena:
        if i == 'U': 
            nivel += 1 # Si i == U Sube un nivel
            if nivel == 0: 
                valles += 1 # Tiene que haber acensos y descensos para poder contar un valle.
        else:
            nivel -= 1 # 
    return valles

pruebas = {
    'DDDUUD' : 0,
    'DDDUUDUU' : 1,
    'dudududududududududu' : 10,
    'DDDUUU' : 1,
    'uuddduuddudu' : 3,
    'u' : 'Error: Out of accepted limits.',
    '' : 'Error: Out of accepted limits.',
    '       DDDUUDUU   ' : 1,
    'dfg34534efbnvmvjjwerwerwerwer' : 0
    }

count = 0

for i in pruebas:
    if valley_counts(i) == pruebas.get(i):
        count += 1
        print(f'Valles: {valley_counts(i)}')

print(f'Total passed tests: {count} out of {len(pruebas)}')

