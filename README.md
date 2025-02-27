# Modelo de Predicción de Precios de Bienes Raíces en Connecticut

***Desarrolladores:***
   * Mariana Lopera Correa
   * Kevin Sebastian Cifuentes Lopez

## Descripción del proyecto
Este proyecto utiliza un modelo de Machine Learning basado en **Random Forest** para predecir los precios de bienes raíces en estado de Connecticut, Estados Unidos. El modelo se entrena con datos de la última década y permite realizar consultas futuras sobre el valor estimado de las propiedades.

## Requisitos

- Python 3.x
- Bibliotecas necesarias:
  - `poetry`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Para la instalación y el correcto funcionamiento del entorno virtual con Poetry siga la siguiente dirección web: [Instalación de Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

Para ejecutar el entorno virtual de Poetry en su dispositivo asegurese que tenga instalado y configurado en el PATH del sistema con el siguiente comando:
```bash
poetry --version
```
***Output:** Poetry (version.number)*

Luego de verificar el correcto funcionamiento del comando *poetry* ejecute el siguiente comando para la creación del entorno virtual en carpeta:
```bash
poetry config virtualenvs.in-project true
```

Ahora instalaremos las dependencias necesarias y crearemos el entorno virtual:
```bash
poetry install
```
***Output:** creará una carptea .venv dentro del proyecto, instalará las dependencias necesarias*

Por último activar el entorno virtual de Poetry:
```bash
eval $(poetry env activate)
```

**Nota final:** es importante que en su Entorno de desarrollo integrado (IDE) tenga activado el interprete del entorno virtual y no el global. Puede buscar más información en la página de ayuda de su IDE. Aquí está para [Visual Studio Code](https://code.visualstudio.com/docs/python/environments#_working-with-python-interpreters).

## Datos

Los datos son sacados de [**Kaggle**](https://www.kaggle.com/datasets/utkarshx27/real-estate-sales-2001-2020-gl).

El dataset utilizado para entrenar y testear el modelo contiene información de ventas de bienes raíces en Connecticut entre los años 2001 y 2022. Para este modelo, se seleccionaron los datos de la última década y para decidir las variables que se usaron en el modelo se realizó una matriz de correlación de Pearson, finalmente, se filtraron las siguientes columnas relevantes:

- **Assessed Value**: Valor tasado de la propiedad.
- **Sale Amount**: Precio de venta.
- **Sales Ratio**: Relación entre el valor tasado y el precio de venta.
- **Property Type**: Tipo de propiedad.
- **Residential Type**: Tipo de vivienda.
- **Location**: Ubicación de la propiedad.

## Proceso de Modelado

1. **Carga y preprocesamiento de datos**:
   - Se filtraron los datos de los últimos 10 años.
   - Se trataron los valores nulos y duplicados.
   - Se convierten variables categóricas en variables numéricas usando *One-Hot Encoding*.
   
2. **División del conjunto de datos**:
   - 80% para entrenamiento.
   - 20% para prueba.

3. **Entrenamiento del modelo**:
   - Se utiliza un **Random Forest Regressor** con 100 árboles.
   - Se normalizan los datos numéricos con `StandardScaler`.
   - Se ajusta el modelo y se evalúa con las métricas **MAE** y **R2-score**.

4. **Predicción futura**:
   - Se pueden ingresar valores de ubicación, tipo de propiedad y otras características para obtener una estimación del precio futuro.

## Casos de Prueba

Se han definido los siguientes tipos de pruebas para garantizar la robustez del modelo:

1. **Casos Normales**:
   - Predicción con datos típicos de propiedades.
   - Evaluación del modelo con datos reales.

2. **Casos Extraordinarios**:
   - Propiedades con valores extremadamente altos o bajos.
   - Ubicaciones poco comunes en el conjunto de datos.

3. **Casos de Error**:
   - Datos faltantes o incorrectos.
   - Tipo de propiedad desconocido.

Estos casos de prueba se encuentran explicados en el siguiente libro de excel: [Casos de Prueba ML](https://1drv.ms/x/c/f683d4d40cda38bf/EfH2jjzjMn9Eg6g6Ur0FsKIBmuThKwrm03PlM_XblCBf5Q?e=kb987y)

---

Este proyecto permite estimar el valor de bienes raíces en Medellín utilizando datos históricos y modelos de Machine Learning. Para mejorar su rendimiento, se recomienda ajustar los hiperparámetros y utilizar técnicas de validación cruzada.


---

## Estructura del Proyecto
El código se compone de los siguientes módulos:

- `modelo.py`: Contiene la implementación del modelo de **Random Forest**.
- `procesamiento.py`: Encargado de la limpieza y transformación de los datos.
- `interfaz.py`: Proporciona la interfaz en consola para interactuar con el usuario.

## Entradas del Programa
El usuario puede ingresar datos en dos formatos:
1. **Archivo CSV**: Conjunto de datos históricos de precios de bienes raíces en Medellín.
2. **Entrada manual en consola**: Permite consultar predicciones ingresando:
   - `Año de consulta`
   - `Barrio de Medellín`
   - `Rango de precios`

## Procesamiento de Datos
1. **Carga de Datos**: Se importan desde un archivo CSV o DataFrame.
2. **Limpieza y Normalización**: Se eliminan datos faltantes y se convierten variables categóricas en numéricas.
3. **Entrenamiento del Modelo**: Se utiliza un conjunto de datos dividido en 80% para entrenamiento y 20% para prueba.
4. **Predicción**: El modelo genera estimaciones basadas en las entradas del usuario.

## Salidas del Programa
El programa proporciona los siguientes tipos de salidas:

1. **Predicciones de Precios**: Devuelve el precio estimado basado en las entradas del usuario.
2. **Mensajes de Advertencia**: Si no se encuentran datos suficientes para una consulta, se informa al usuario.
3. **Errores de Entrada**: Manejo de errores como años fuera del rango de datos o barrios inexistentes.


## Uso del Programa
Para ejecutar la interfaz en consola, utilice:
```bash
python interfaz.py
```

El usuario podrá seleccionar opciones para realizar consultas sobre precios de bienes raíces en Medellín de acuerdo con los datos ingresados.

## Contacto
Para más información o contribuciones, comuníquese con el equipo de desarrollo.

