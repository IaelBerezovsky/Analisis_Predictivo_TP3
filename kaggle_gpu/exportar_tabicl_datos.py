"""Exports the 147 variables of model 15b to kaggle/tabicl_datos/ (parquet) for TabICLv2 on Kaggle.

Nested version (evaluation): for each outer fold k, train = rows of the other 4 folds and valid = rows of
fold k, with the target-based variables of that outer fold (predicciones/anidadas/), exactly as the
LightGBM 15b used them in evaluar_modelo_anidado. Fine-tunings use their standard version (as in models 13 and 15).
Standard version (test): full train with out-of-fold target variables and test computed with all of train.

Run from the project root: APTP3.venv/Scripts/python.exe kaggle/exportar_tabicl_datos.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from features import (agregar_features_duracion, agregar_features_fechas, agregar_features_texto,
                      leer_fotos_atributos, leer_json_crudo, valores_campos_json, variables_atributos,
                      variables_fotos, variables_patrones, variables_vendedor_categoria)
from modelado import (ARCHIVO_TEST, CARPETA_PREDICCIONES, cargar_anidadas, combinar_anidadas, obtener_folds,
                      preparar_datos)

inicio = time.time()
SALIDA = Path("kaggle/tabicl_datos")
SALIDA.mkdir(parents=True, exist_ok=True)
with open("parametros_lgbm_14.json", encoding="utf-8") as f:
    PARAMETROS_14 = json.load(f)

# Same construction as modelo_15_combinacion_final.ipynb (without the raw ids, which 15b does not use)
X, y, X_test, ids_test, df_train, df_test = preparar_datos(
    agregar_features=lambda df: agregar_features_duracion(agregar_features_fechas(agregar_features_texto(df))),
    devolver_procesados=True)
variables_train, variables_test = variables_vendedor_categoria(df_train, df_test)
X, X_test = X.join(variables_train), X_test.join(variables_test)
datos_train = leer_fotos_atributos("train_data.jsonlines")
datos_test = leer_fotos_atributos(ARCHIVO_TEST)
datos_train.index, datos_test.index = X.index, X_test.index
atributos_train, atributos_test = variables_atributos(datos_train, datos_test)
X = X.join(variables_fotos(datos_train)).join(atributos_train)
X_test = X_test.join(variables_fotos(datos_test)).join(atributos_test)
PATRONES = [tuple(p.split(": ", 1)) for p in PARAMETROS_14["patrones"]]
crudo_train, crudo_test = leer_json_crudo("train_data.jsonlines"), leer_json_crudo(ARCHIVO_TEST)
X = X.join(variables_patrones(valores_campos_json(crudo_train), PATRONES).set_index(X.index))
X_test = X_test.join(variables_patrones(valores_campos_json(crudo_test), PATRONES).set_index(X_test.index))

anidadas_texto, *estandar_texto = cargar_anidadas("submodelos_texto")
anidadas_te, *estandar_te = cargar_anidadas("target_encoding_m3")
anidadas = combinar_anidadas(anidadas_texto, anidadas_te)
ESTANDAR_TRAIN = pd.concat([estandar_texto[0], estandar_te[0]], axis=1).set_index(X.index)
ESTANDAR_TEST = pd.concat([estandar_texto[1], estandar_te[1]], axis=1).set_index(X_test.index)

folds = obtener_folds(y)
FINETUNES = {"proba_used_finetune": "titulo", "proba_used_finetune_titulo_warranty": "beto_titulo_warranty",
             "proba_used_finetune_cased": "beto_cased_titulo"}
for variable, archivo in FINETUNES.items():
    oof = pd.read_csv(CARPETA_PREDICCIONES / "finetune" / f"finetune_{archivo}_oof.csv").sort_values("fila")
    test = pd.read_csv(CARPETA_PREDICCIONES / "finetune" / f"finetune_{archivo}_test.csv").sort_values("fila")
    assert (oof["fila"].to_numpy() == np.arange(len(X))).all() and (test["fila"].to_numpy() == np.arange(len(X_test))).all()
    for k, (_, idx_val) in enumerate(folds, start=1):
        assert (np.sort(idx_val) == np.flatnonzero(oof["fold"].to_numpy() == k)).all(), f"{archivo}: fold {k}"
    X[variable], X_test[variable] = oof["proba_used"].to_numpy(), test["proba_used"].to_numpy()
X["proba_used_finetune_promedio"] = X[list(FINETUNES)].mean(axis=1)
X_test["proba_used_finetune_promedio"] = X_test[list(FINETUNES)].mean(axis=1)

# Columns of 15b: 13c without proba_used_finetune, plus the average of the three fine-tunings
COLUMNAS = [c for c in PARAMETROS_14["columnas"] if c != "proba_used_finetune"] + ["proba_used_finetune_promedio"]
assert len(COLUMNAS) == 147 and len(set(COLUMNAS)) == 147 and not {"y", "fila"} & set(COLUMNAS)
importancia = pd.read_pickle(CARPETA_PREDICCIONES / "modelo_15" / "lgbm_b_estandar.pkl")["importancia"]
assert set(importancia.index) == set(COLUMNAS)
TOP100 = list(importancia.sort_values("importancia", ascending=False).index[:100])


def a_exportar(datos):
    # Numeric (including booleans) to float32; categoricals keep their fixed train categories; NaN stay NaN
    datos = datos[COLUMNAS].copy()
    for c in COLUMNAS:
        if not isinstance(datos[c].dtype, pd.CategoricalDtype):
            assert datos[c].dtype.kind in "biuf", f"{c}: unexpected dtype {datos[c].dtype}"
            datos[c] = datos[c].astype("float32")
    return datos.reset_index(drop=True)


def con_y_fila(datos, idx):
    datos = a_exportar(datos.iloc[idx])
    datos["y"] = y.to_numpy()[idx].astype("int8")
    datos["fila"] = np.asarray(idx, dtype="int64")
    return datos


Y = y.to_numpy()
X_anidadas = {}
for k, (idx_tr, idx_val) in enumerate(folds, start=1):
    X_k = X.join(anidadas[k - 1].set_index(X.index))  # same join as evaluar_modelo_anidado
    X_anidadas[k] = X_k
    con_y_fila(X_k, idx_tr).to_parquet(SALIDA / f"anidada_fold{k}_train.parquet", index=False)
    con_y_fila(X_k, idx_val).to_parquet(SALIDA / f"anidada_fold{k}_valid.parquet", index=False)
X_std, X_test_std = X.join(ESTANDAR_TRAIN), X_test.join(ESTANDAR_TEST)
con_y_fila(X_std, np.arange(len(X_std))).to_parquet(SALIDA / "estandar_train.parquet", index=False)
test_exportado = a_exportar(X_test_std)
test_exportado["fila"] = np.arange(len(X_test_std), dtype="int64")
test_exportado.to_parquet(SALIDA / "estandar_test.parquet", index=False)

CATEGORICAS = [c for c in COLUMNAS if isinstance(X_std[c].dtype, pd.CategoricalDtype)]
with open(SALIDA / "columnas.json", "w", encoding="utf-8") as f:
    json.dump({"descripcion": "Variables de la variante 15b (08 + promedio de los tres fine-tunings + 22 patrones). "
                              "top100 = las 100 de mayor ganancia promedio del LightGBM 15b (corrida estándar).",
               "todas": COLUMNAS, "top100": TOP100, "categoricas": CATEGORICAS,
               # pyarrow (Kaggle) reads fastparquet categoricals as str: the notebook restores these train categories
               "categorias": {c: [str(v) for v in X_std[c].cat.categories] for c in CATEGORICAS},
               "lgbm_15b_accuracy_anidada": 0.94583}, f, ensure_ascii=False, indent=1)

# ---- Verification ----
print(f"Columnas: {len(COLUMNAS)} ({len(CATEGORICAS)} categóricas: {CATEGORICAS}); top100: {len(TOP100)}")
TARGET = list(anidadas[0].columns)
print(f"Variables con target ({len(TARGET)}): {TARGET}")
for k, (idx_tr, idx_val) in enumerate(folds, start=1):
    for parte, idx in [("train", idx_tr), ("valid", idx_val)]:
        leido = pd.read_parquet(SALIDA / f"anidada_fold{k}_{parte}.parquet")
        assert list(leido.columns) == COLUMNAS + ["y", "fila"]
        assert (leido["fila"].to_numpy() == idx).all() and (leido["y"].to_numpy() == Y[idx]).all()
        guardado = anidadas[k - 1].iloc[idx]  # what predicciones/anidadas/ has for this outer fold
        for c in TARGET:
            if c in COLUMNAS:
                np.testing.assert_allclose(leido[c].to_numpy(dtype=float), guardado[c].to_numpy(dtype=float),
                                           rtol=1e-6, atol=1e-7, equal_nan=True, err_msg=f"fold {k} {parte} {c}")
        for c in CATEGORICAS:
            assert list(leido[c].cat.categories) == list(X_std[c].cat.categories), c
            assert leido[c].astype(object).equals(X_anidadas[k][c].iloc[idx].astype(object).reset_index(drop=True)), c
print("Anidada: las variables con target de cada fold coinciden con predicciones/anidadas/ (float32) "
      "y las categóricas conservan las categorías de train")

# Fold 1 of the LightGBM 15b retrained from the parquet files, against the saved nested predictions
PARAMS = PARAMETROS_14["parametros_lgbmclassifier"]
tr = pd.read_parquet(SALIDA / "anidada_fold1_train.parquet")
va = pd.read_parquet(SALIDA / "anidada_fold1_valid.parquet")
modelo = lgb.LGBMClassifier(**PARAMS, random_state=42, bagging_seed=42)
modelo.fit(tr[COLUMNAS], tr["y"], eval_X=va[COLUMNAS], eval_y=va["y"],
           callbacks=[lgb.early_stopping(PARAMETROS_14["early_stopping"], verbose=False)])
proba = modelo.predict_proba(va[COLUMNAS])[:, 1]
guardada = pd.read_pickle(CARPETA_PREDICCIONES / "modelo_15" / "lgbm_b_anidada.pkl")["oof"].to_numpy()[va["fila"]]
print(f"LightGBM 15b fold 1 desde parquet: accuracy {accuracy_score(va['y'], proba >= 0.5):.4f} "
      f"(guardado {accuracy_score(va['y'], guardada >= 0.5):.4f}), correlación {np.corrcoef(proba, guardada)[0, 1]:.5f}, "
      f"máxima diferencia de probabilidad {np.abs(proba - guardada).max():.4f}, "
      f"predicciones distintas {int(((proba >= 0.5) != (guardada >= 0.5)).sum())} de {len(proba)}")

leido = pd.read_parquet(SALIDA / "estandar_train.parquet")
for c in ESTANDAR_TRAIN.columns:
    if c in COLUMNAS:
        np.testing.assert_allclose(leido[c].to_numpy(dtype=float), ESTANDAR_TRAIN[c].to_numpy(dtype=float),
                                   rtol=1e-6, atol=1e-7, equal_nan=True)
leido_test = pd.read_parquet(SALIDA / "estandar_test.parquet")
assert list(leido_test.columns) == COLUMNAS + ["fila"] and len(leido_test) == 30000 and len(leido) == 70000
print("Estándar: train (70.000) y test (30.000) verificados")

total = sum(p.stat().st_size for p in SALIDA.iterdir())
for p in sorted(SALIDA.iterdir()):
    print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")
print(f"Tamaño total: {total / 1e6:.1f} MB | {(time.time() - inicio) / 60:.1f} min")
