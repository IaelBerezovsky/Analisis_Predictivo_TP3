# Condición de publicaciones de MercadoLibre: nuevo o usado

Competencia de Kaggle (TP3): predecir si una publicación de MercadoLibre de 2015 es de un producto nuevo (`new`) o usado (`used`) a partir de sus datos (título, precio, stock, vendedor, categoría, fotos, atributos, fechas...). Métrica: accuracy.

| Validación anidada (5 folds) | Kaggle público | Kaggle privado | Puesto |
|---|---|---|---|
| 0,9473 | 0,94690 | 0,94844 | 11 / 32 |

## Cómo reproducir

1. Descargar `train_data.jsonlines` y `test_data.jsonlines` de la competencia en Kaggle y copiarlos en `datos/`.
2. Instalar las dependencias (Python 3.13):
   ```
   pip install -r requirements.txt
   ```
3. Correr, desde la carpeta del repo, `1_entrenamiento.ipynb` (~85 min en CPU): arma las variables, entrena LightGBM y CatBoost, elige los pesos del promedio y guarda los modelos en `modelos/`.
4. Correr `2_prediccion.ipynb` (~2-5 min): rearma las variables de test, aplica los modelos guardados y escribe `submission/prediccion_reproducida.csv`. Verifica que coincide fila por fila con la submission enviada (`submission/19_atributos_categoria_titulo.csv`).

El paso 4 se puede correr solo, sin el 3: los modelos ya están guardados en `modelos/`.

## El modelo final

```
jsonlines ──► procesar() + variables sin target ─────────────────────────┐
              (texto, fechas, duración, vendedor y categoría, fotos,     │
               atributos, 22 patrones, categoría descrita por títulos)   │
                                                                          │
              variables con target, out-of-fold ─────────────────────────┤
              (target encoding de vendedor y categoría, TF-IDF + logística│
               del título y la garantía, % used de categorías vecinas)   │
                                                                          │
              fine-tunings de BETO (GPU, promedio de 3) ─────────────────┤
                                                                          ▼
                   ┌───────────────┬───────────────────────┬───────────────────┐
                   │ LightGBM      │ CatBoost con ids      │ TabICLv2          │
                   │ 164 variables │ crudos, 123 variables │ top 100 (GPU)     │
                   └───────┬───────┴───────────┬───────────┴─────────┬─────────┘
                           └───────────► promedio ponderado ◄────────┘
                                               │
                                          umbral 0,5 ──► new / used
```

| Modelo | Accuracy anidada | Peso |
|---|---|---|
| LightGBM | 0,9467 | 0,624 |
| CatBoost con ids crudos (`seller_id`, `category_id`, ciudad) | 0,9434 | 0,175 |
| TabICLv2 (100 variables más importantes del LightGBM) | 0,9435 | 0,201 |
| **Promedio ponderado** | **0,9473** | |

- **Validación:** 5 folds estratificados (`random_state=42`), siempre los mismos. Las variables que usan el target se calculan out-of-fold. Para evaluar se usa una CV anidada: en cada fold, esas variables salen solo de su entrenamiento (con folds internos).
- **Pesos:** minimizan el log loss de las predicciones out-of-fold anidadas, con pesos no negativos que suman 1.
- **Predicción de test:** para LightGBM y CatBoost, es el promedio de los 5 modelos de la CV (uno por fold), no un modelo reentrenado con todo train.

## Partes entrenadas en GPU

Tres fine-tunings de BETO (`dccuchile/bert-base-spanish-wwm-uncased` sobre el título y sobre título + garantía; `-cased` sobre el título) y TabICLv2 se entrenaron en Kaggle con GPU, con los mismos 5 folds. Los fine-tunings dan una variable del LightGBM (el promedio de los tres); TabICLv2 es el tercer miembro del promedio.

No se reentrenan: el entrenamiento en GPU no es determinístico bit a bit (sumas en paralelo cuyo orden cambia entre corridas), así que repetirlo cambiaría algunas probabilidades y algunas filas del resultado. Entran como predicciones guardadas (out-of-fold de train y de test) en `predicciones_gpu/`.

Los notebooks de Kaggle están en `kaggle_gpu/` como referencia, con `folds.csv` y las instrucciones (`INSTRUCCIONES_KAGGLE.md`). `exportar_tabicl_datos.py` genera los datos de entrada de TabICL y depende de la estructura del proyecto original.

## Estructura

```
├── 1_entrenamiento.ipynb     entrenamiento completo (CV anidada y estándar, pesos, guardado)
├── 2_prediccion.ipynb        predicción de test con los modelos guardados + verificación
├── src/
│   ├── procesamiento.py      carga de los jsonlines y limpieza (procesar)
│   ├── features.py           variables sin target
│   ├── modelado.py           folds, variables con target, CV, pesos y submission
│   └── configuracion.json    columnas, hiperparámetros y patrones del modelo final
├── datos/                    jsonlines de Kaggle (no se suben)
├── predicciones_gpu/         predicciones de los modelos entrenados en GPU
├── kaggle_gpu/               notebooks de Kaggle (referencia)
├── modelos/                  5 LightGBM (.txt), 5 CatBoost (.cbm), pesos y matrices de test para verificar
├── submission/               submission enviada y la reproducida
└── requirements.txt
```
