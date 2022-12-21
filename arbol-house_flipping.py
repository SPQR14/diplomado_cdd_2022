

def crea_variable_objetivo(price=0, view=0, grade=0, condition=0, sqft_living=0, yr_renovated=0, yr_built=0, waterfront=0, profitability=0):
    """
    Esta función devuelve un valor booleano (0 o 1) a partir del cumplimiento o no de una serie de condiciones que se tienen que cumplir según el criterio de
    elegibilidad para adquirir una propiedad para maximizar las ganancias en un esquema de "House Flipping".

    Args:
        price (int, mandatory): El precio de la propiedad, tiene un peso de 4 . Defaults to 0.
        view (int, mandatory): La calificación de la vista desde la propiedad, tiene un peso asignado de 3. Defaults to 0.
        grade (int, mandatory): Una calificación general asignada a la propiedad en la que se cuenta el diseño y la calidad de la construcción
        su peso asignado es de 4. Defaults to 0.
        condition (int, mandatory): Condición de la propiedad, tiene un peso de 3. Defaults to 0.
        sqft_living (int, mandatory): Tamaño en pues cuadrados de la superficie habitable de la propiedad, tiene un peso de 4. Defaults to 0.
        yr_renovated (int, mandatory): Año desde la última remodelación, tiene un peso de 3. Defaults to 0.
        yr_built (int, mandatory): Año de la construcción, tiene un peso de 4. Defaults to 0.
        waterfront (bool, mandatory): Variable que indica si la propiedad cuenta con vista al mar. Defaults to 0.
        profitability (double, mandatory): Variable calculada que indica el índice de rentabilidad de una propiedad, tiene un peso de 5. Defaults to 0.
    """

    # Threshold definition
    median_price = 450000
    median_view = 2
    grade_l = 6
    grade_h = 10
    condition_ = 3
    median_sqft = 1910
    currrent_year = 2022

    profitability_l = (0.1131 - 0.046)
    profitability_h = (0.1131 + 0.046)

    total_index = 0

    # Desition tree
    if price <= median_price:
        total_index += 4
    else:
        total_index += 0


    if view > median_view:
        total_index += 3
    else:
        total_index += 0
    

    if grade < grade_l:
        total_index += 0
    elif grade >= grade_l and grade < grade_h:
        total_index += 2
    elif grade >= grade_h:
        total_index += 4


    if condition >= condition_:
        total_index += 3
    else:
        total_index += 0
    

    if sqft_living >= median_sqft:
        total_index += 4
    else:
        total_index += 0
    

    if (currrent_year - yr_renovated) < 10:
        total_index += 3
    elif (currrent_year - yr_renovated) >= 10:
        total_index += 1
    elif yr_renovated == 0:
        total_index += 0

    

    if profitability < profitability_l:
        total_index += 0
    elif profitability >= profitability_l and profitability < profitability_h:
        total_index += 3
    elif profitability >= profitability_h:
        total_index += 5

    
    if waterfront == 1:
        total_index += 3
    else:
        total_index += 0
    

    if(currrent_year - yr_built) < 30:
        total_index += 4
    elif(currrent_year - yr_built) >= 30 and (currrent_year - yr_built) < 60:
        total_index += 2
    else:
        total_index += 0


    if total_index >= 20:
        return 1
    else:
        return 0
    pass


des = crea_variable_objetivo(price=550000, view=5, grade=10, condition=5,  sqft_living=3000, yr_renovated=2014, yr_built=1930, profitability=6)

print(des)