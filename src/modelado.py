"""Funciones comunes del modelo final (copiadas sin cambios de modelado.py del proyecto; solo las que usa el modelo 19).

Cambios respecto del original, solo de organización:
- rutas: los datos se leen de datos/ y preparar_datos() le pasa esas rutas a cargar_datos();
- guardar_submission() escribe en submission/;
- ajustar_pesos() (notebook del modelo 19) y como_texto() (notebook del modelo 10) pasaron a este archivo.

Todos los modelos usan los mismos folds (obtener_folds), así las predicciones out-of-fold se pueden combinar.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from procesamiento import cargar_datos, procesar

CARPETA = Path(__file__).resolve().parent.parent
CARPETA_DATOS = CARPETA / "datos"
ARCHIVO_TRAIN = CARPETA_DATOS / "train_data.jsonlines"
ARCHIVO_TEST = CARPETA_DATOS / "test_data.jsonlines"
CARPETA_GPU = CARPETA / "predicciones_gpu"
CARPETA_MODELOS = CARPETA / "modelos"
CARPETA_SUBMISSIONS = CARPETA / "submission"
ARCHIVO_CONFIGURACION = Path(__file__).resolve().parent / "configuracion.json"

SEMILLA = 42
N_FOLDS = 5
FILAS_TEST = 30000
CLASES = {0: "new", 1: "used"}

# Formato de la submission de Kaggle: columnas ID y condition (new/used). ID es el número de fila de test
# en el orden de test_data.jsonlines (no el id MLA de la publicación), empezando en INICIO_ID.
COLUMNA_ID_SUBMISSION = "ID"
COLUMNA_TARGET_SUBMISSION = "condition"
INICIO_ID = 1

# Columnas que nunca son variables explicativas
COLUMNAS_SIEMPRE_FUERA = ["condition", "id"]

# Exclusiones del modelo base
EXCLUIR_POR_DEFECTO = [
    # texto libre
    "title", "warranty", "permalink",
    # fechas
    "start_time", "stop_time", "date_created", "last_updated",
    # alta cardinalidad
    "seller_id", "category_id", "seller_address_city_name",
    # duplicados exactos de otra columna
    "seller_address_state_id", "shipping_free_methods_cantidad", "atributos_grupo_rangcuota",
]


def cargar_configuracion():
    """Columnas (en el orden original), parámetros y patrones del modelo final (src/configuracion.json)."""
    with open(ARCHIVO_CONFIGURACION, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------------------------------
def preparar_datos(excluir=None, agregar_features=None, devolver_procesados=False):
    """Carga train y test con cargar_datos() y procesar() y arma las matrices para modelar.

    Parámetros
    ----------
    excluir : list, opcional
        Columnas que no entran como variables explicativas. Por defecto EXCLUIR_POR_DEFECTO.
        condition e id se excluyen siempre. Para sumar una columna excluida, por ejemplo:
        excluir=[c for c in EXCLUIR_POR_DEFECTO if c != "category_id"].
    agregar_features : callable, opcional
        Función df -> df que se aplica a train y a test ya procesados, antes de elegir las
        columnas. Sirve para el feature engineering sin modificar procesar().
    devolver_procesados : bool
        Si es True, devuelve además df_train y df_test procesados con todas sus columnas (por
        ejemplo, para usar title o warranty en un submodelo con variable_oof).

    Devuelve
    --------
    X_train, y, X_test, ids_test (y df_train, df_test si devolver_procesados=True)
        y vale 1 para used y 0 para new. Las booleanas pasan a int, las category quedan como
        category y las columnas de texto que queden se pasan a category con las categorías de
        train (un valor de test que no esté en train queda como NaN).
    """
    excluir = EXCLUIR_POR_DEFECTO if excluir is None else excluir
    df_train, df_test = cargar_datos(ARCHIVO_TRAIN, ARCHIVO_TEST)
    df_train, df_test = procesar(df_train), procesar(df_test)
    if agregar_features is not None:
        df_train, df_test = agregar_features(df_train), agregar_features(df_test)

    y = (df_train["condition"] == "used").astype(int).rename("used")
    ids_test = df_test["id"].reset_index(drop=True)

    fuera = set(COLUMNAS_SIEMPRE_FUERA) | set(excluir)
    columnas = [c for c in df_train.columns if c not in fuera]
    faltan_en_test = [c for c in columnas if c not in df_test.columns]
    if faltan_en_test:
        raise ValueError(f"Columnas de train que no están en test: {faltan_en_test}")
    X_train, X_test = df_train[columnas].copy(), df_test[columnas].copy()

    for col in X_train.select_dtypes(include="bool").columns:
        X_train[col], X_test[col] = X_train[col].astype(int), X_test[col].astype(int)
    for col in X_train.select_dtypes(include=["str", "object"]).columns:
        tipo = pd.CategoricalDtype(sorted(X_train[col].dropna().unique()))
        X_train[col], X_test[col] = X_train[col].astype(tipo), X_test[col].astype(tipo)
    for col in X_train.select_dtypes(include="category").columns:
        if not X_train[col].cat.categories.equals(X_test[col].cat.categories):
            X_test[col] = X_test[col].astype(X_train[col].dtype)
    if devolver_procesados:
        return X_train, y, X_test, ids_test, df_train, df_test
    return X_train, y, X_test, ids_test


# ---------------------------------------------------------------------------------------------------
# Validación cruzada
# ---------------------------------------------------------------------------------------------------
def obtener_folds(y):
    """Los mismos 5 folds estratificados para todos los modelos: lista de (idx_train, idx_val)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    return list(skf.split(np.zeros(len(y)), y))


def evaluar_modelo(crear_modelo, X, y, X_test, fit_params=None, umbral=0.5):
    """Entrena un modelo por fold y evalúa con los folds de obtener_folds().

    Parámetros
    ----------
    crear_modelo : callable
        Función sin argumentos que devuelve un modelo nuevo con fit() y predict_proba()
        (sklearn, LightGBM, XGBoost, CatBoost...).
    fit_params : dict o callable, opcional
        Argumentos extra para fit(). Si es una función, recibe (X_val, y_val) del fold y devuelve
        el dict: sirve para early stopping sobre el fold de validación.
    umbral : float
        Umbral de probabilidad de used para calcular la accuracy.

    Devuelve
    --------
    dict con: accuracy_folds, accuracy_media, accuracy_desvio, auc_folds, auc (sobre todas las
    predicciones out-of-fold), oof (probabilidad de used out-of-fold), proba_test (promedio de los
    folds), importancia (DataFrame o None), iteraciones (si el modelo usó early stopping) y modelos.
    """
    folds = obtener_folds(y)
    oof = pd.Series(np.nan, index=X.index, name="proba_used")
    proba_test = np.zeros(len(X_test))
    accuracy_folds, auc_folds, iteraciones, importancias, modelos = [], [], [], [], []

    for n, (idx_tr, idx_val) in enumerate(folds, start=1):
        X_tr, X_val = X.iloc[idx_tr], X.iloc[idx_val]
        y_tr, y_val = y.iloc[idx_tr], y.iloc[idx_val]
        params = fit_params(X_val, y_val) if callable(fit_params) else (fit_params or {})

        modelo = crear_modelo()
        modelo.fit(X_tr, y_tr, **params)

        proba_val = modelo.predict_proba(X_val)[:, 1]
        oof.iloc[idx_val] = proba_val
        proba_test += modelo.predict_proba(X_test)[:, 1] / len(folds)

        accuracy_folds.append(accuracy_score(y_val, proba_val >= umbral))
        auc_folds.append(roc_auc_score(y_val, proba_val))
        iteraciones.append(getattr(modelo, "best_iteration_", None))
        importancias.append(_importancia(modelo, X.columns))
        modelos.append(modelo)
        print(f"Fold {n}: accuracy = {accuracy_folds[-1]:.4f} | AUC = {auc_folds[-1]:.4f}"
              + (f" | iteraciones = {iteraciones[-1]}" if iteraciones[-1] else ""))

    importancia = None
    if all(i is not None for i in importancias):
        importancia = (pd.concat(importancias, axis=1).mean(axis=1)
                       .sort_values(ascending=False).rename("importancia").to_frame())
        importancia["importancia_pct"] = importancia["importancia"] / importancia["importancia"].sum() * 100

    resultado = {
        "accuracy_folds": accuracy_folds,
        "accuracy_media": float(np.mean(accuracy_folds)),
        "accuracy_desvio": float(np.std(accuracy_folds, ddof=1)),
        "auc_folds": auc_folds,
        "auc": float(roc_auc_score(y, oof)),
        "accuracy_oof": float(accuracy_score(y, oof >= umbral)),
        "oof": oof,
        "proba_test": proba_test,
        "importancia": importancia,
        "iteraciones": iteraciones if any(iteraciones) else None,
        "umbral": umbral,
        "modelos": modelos,
    }
    print(f"Accuracy CV: {resultado['accuracy_media']:.4f} ± {resultado['accuracy_desvio']:.4f} | "
          f"AUC CV: {resultado['auc']:.4f}")
    return resultado


def resultado_desde_oof(oof, proba_test, y, umbral=0.5):
    """Arma un resultado con el mismo formato que evaluar_modelo() a partir de predicciones out-of-fold ya
    calculadas (submodelos, promedios, stacking), con accuracy y AUC por fold de obtener_folds(y)."""
    oof = pd.Series(np.asarray(oof, dtype=float), index=pd.Series(y).index, name="proba_used")
    folds = obtener_folds(y)
    accuracy_folds = [accuracy_score(y.iloc[idx], oof.iloc[idx] >= umbral) for _, idx in folds]
    return {
        "accuracy_folds": accuracy_folds,
        "accuracy_media": float(np.mean(accuracy_folds)),
        "accuracy_desvio": float(np.std(accuracy_folds, ddof=1)),
        "auc_folds": [roc_auc_score(y.iloc[idx], oof.iloc[idx]) for _, idx in folds],
        "auc": float(roc_auc_score(y, oof)),
        "accuracy_oof": float(accuracy_score(y, oof >= umbral)),
        "oof": oof,
        "proba_test": np.asarray(proba_test, dtype=float),
        "importancia": None,
        "iteraciones": None,
        "umbral": umbral,
        "modelos": [],
    }


def _importancia(modelo, columnas):
    if hasattr(modelo, "feature_importances_"):
        return pd.Series(modelo.feature_importances_, index=columnas, dtype=float)
    if hasattr(modelo, "coef_"):
        return pd.Series(np.abs(np.ravel(modelo.coef_)), index=columnas, dtype=float)
    return None


# ---------------------------------------------------------------------------------------------------
# Validación anidada: variables con target calculadas solo con las filas de ajuste
# ---------------------------------------------------------------------------------------------------
def obtener_folds_internos(idx_entrenamiento, y):
    """Folds internos (StratifiedKFold de 5, random_state=42) dentro de las filas de entrenamiento de un fold
    externo. Devuelve posiciones de train (no posiciones dentro de idx_entrenamiento)."""
    idx_entrenamiento = np.asarray(idx_entrenamiento)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    return [(idx_entrenamiento[a], idx_entrenamiento[b])
            for a, b in skf.split(np.zeros(len(idx_entrenamiento)), np.asarray(y)[idx_entrenamiento])]


def variables_anidadas(calcular, y):
    """Variables con target para la validación anidada: dict fold externo -> DataFrame con todas las filas de train.

    Para el fold externo k, las filas de validación usan lo ajustado con todo el entrenamiento externo, y las de
    entrenamiento externo se calculan out-of-fold con folds internos. La fold de validación nunca entra en el
    cálculo, igual que pasa con test.
    """
    resultado = {}
    for k, (idx_tr, idx_val) in enumerate(obtener_folds(y)):
        partes = [calcular(idx_tr, idx_val)]
        partes += [calcular(idx_tr_int, idx_val_int) for idx_tr_int, idx_val_int in obtener_folds_internos(idx_tr, y)]
        resultado[k] = pd.concat(partes).sort_index()
    return resultado


def variables_estandar(calcular, y):
    """Variables con target para la submission: train out-of-fold con obtener_folds(y) y test con todo train.
    Devuelve (DataFrame de train, DataFrame de test)."""
    train = pd.concat([calcular(idx_tr, idx_val) for idx_tr, idx_val in obtener_folds(y)]).sort_index()
    return train, calcular(np.arange(len(y)), None)


def combinar_anidadas(*anidadas):
    """Junta por columnas varios resultados de variables_anidadas (un DataFrame por fold externo)."""
    return {k: pd.concat([a[k] for a in anidadas], axis=1) for k in anidadas[0]}


def calculador_target_encoding(columna_train, columna_test, y, nombre, m=10):
    """Calculador de te_{nombre} (% de used suavizado, como target_encoding_oof) y conteo_{nombre}_train."""
    columna_train = pd.Series(columna_train).reset_index(drop=True)
    columna_test = pd.Series(columna_test).reset_index(drop=True)
    y_ = pd.Series(np.asarray(y))

    def calcular(idx_ajuste, idx_aplicar):
        prior = y_.iloc[idx_ajuste].mean()
        stats = y_.iloc[idx_ajuste].groupby(columna_train.iloc[idx_ajuste].values).agg(["sum", "size"])
        encoding = (stats["sum"] + m * prior) / (stats["size"] + m)
        aplicar = columna_test if idx_aplicar is None else columna_train.iloc[idx_aplicar]
        indice = np.arange(len(columna_test)) if idx_aplicar is None else idx_aplicar
        return pd.DataFrame({f"te_{nombre}": aplicar.map(encoding).fillna(prior).to_numpy(dtype=float),
                             f"conteo_{nombre}_train": aplicar.map(stats["size"]).fillna(0).to_numpy(dtype=float)},
                            index=indice)
    return calcular


def calculador_submodelo(crear_modelo, entrada_train, entrada_test, y, nombre):
    """Calculador de la probabilidad de used de un submodelo (por ejemplo TF-IDF + logística sobre el título)."""
    y_ = pd.Series(np.asarray(y))

    def filas(datos, idx):
        return datos.iloc[idx] if hasattr(datos, "iloc") else datos[idx]

    def calcular(idx_ajuste, idx_aplicar):
        modelo = crear_modelo()
        modelo.fit(filas(entrada_train, idx_ajuste), y_.iloc[idx_ajuste])
        aplicar = entrada_test if idx_aplicar is None else filas(entrada_train, idx_aplicar)
        indice = np.arange(len(entrada_test)) if idx_aplicar is None else idx_aplicar
        return pd.DataFrame({nombre: modelo.predict_proba(aplicar)[:, 1]}, index=indice)
    return calcular


def calculador_te_categorias_vecinas(categoria_train, categoria_test, vecinas, y, nombre="categorias_vecinas", m=3):
    """Calculador de te_{nombre}: % de used de las categorías vecinas (modelo 19).

    vecinas: DataFrame indexado por category_id (como texto) con "vecinas" (array de category_id) y "similitudes",
    de features.categorias_por_titulos. Para una categoría con vecinas v_i y similitudes s_i:
        te = (sum s_i * used_i + m * prior) / (sum s_i * n_i + m)
    donde used_i y n_i son las publicaciones used y totales de v_i en las filas de ajuste: cada vecina pesa por
    su similitud y por su cantidad de publicaciones, y el resultado se suaviza hacia el prior de las filas de
    ajuste. La propia categoría nunca entra. Una categoría sin documento queda NaN.
    """
    codigos = {c: i for i, c in enumerate(vecinas.index)}
    cod_train = pd.Series(categoria_train).astype(str).map(codigos).to_numpy(dtype=float)
    cod_test = pd.Series(categoria_test).astype(str).map(codigos).to_numpy(dtype=float)
    vec = np.vstack([[codigos[c] for c in v] for v in vecinas["vecinas"]])
    sim = np.vstack(vecinas["similitudes"].to_numpy())
    y_ = np.asarray(y, dtype=float)
    n_cat = len(codigos)

    def calcular(idx_ajuste, idx_aplicar):
        idx_ajuste = np.asarray(idx_ajuste)
        cod = cod_train[idx_ajuste]
        valido = ~np.isnan(cod)
        n = np.bincount(cod[valido].astype(int), minlength=n_cat)
        used = np.bincount(cod[valido].astype(int), weights=y_[idx_ajuste][valido], minlength=n_cat)
        prior = y_[idx_ajuste].mean()
        te_categoria = ((sim * used[vec]).sum(axis=1) + m * prior) / ((sim * n[vec]).sum(axis=1) + m)
        aplicar = cod_test if idx_aplicar is None else cod_train[idx_aplicar]
        indice = np.arange(len(cod_test)) if idx_aplicar is None else np.asarray(idx_aplicar)
        te = np.full(len(aplicar), np.nan)
        con = ~np.isnan(aplicar)
        te[con] = te_categoria[aplicar[con].astype(int)]
        return pd.DataFrame({f"te_{nombre}": te}, index=indice)
    return calcular


def submodelo_tfidf(C):
    """El submodelo de texto de los modelos 02 a 04: TF-IDF de palabras (1-2, min_df=2) + caracteres char_wb
    (2-5, min_df=3), sublinear, y regresión logística (max_iter=2000). Devuelve una función que crea el modelo."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    return lambda: make_pipeline(
        make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
        LogisticRegression(C=C, max_iter=2000),
    )


def evaluar_modelo_anidado(crear_modelo, X, anidadas, y, columnas, fit_params=None, umbral=0.5):
    """Como evaluar_modelo, pero con las variables con target de la validación anidada.

    X tiene las variables sin target; anidadas es el dict fold externo -> DataFrame de variables_anidadas.
    En el fold k se entrena y se evalúa con X + anidadas[k] (solo las `columnas`). No predice test: la
    submission se arma con las variables estándar y evaluar_modelo. Devuelve el mismo formato (proba_test = None).
    """
    oof = pd.Series(np.nan, index=X.index, name="proba_used")
    accuracy_folds, auc_folds, iteraciones, importancias = [], [], [], []
    for n, (idx_tr, idx_val) in enumerate(obtener_folds(y), start=1):
        X_k = X.join(anidadas[n - 1].set_index(X.index))[columnas]
        X_tr, X_val, y_tr, y_val = X_k.iloc[idx_tr], X_k.iloc[idx_val], y.iloc[idx_tr], y.iloc[idx_val]
        params = fit_params(X_val, y_val) if callable(fit_params) else (fit_params or {})
        modelo = crear_modelo()
        modelo.fit(X_tr, y_tr, **params)
        proba_val = modelo.predict_proba(X_val)[:, 1]
        oof.iloc[idx_val] = proba_val
        accuracy_folds.append(accuracy_score(y_val, proba_val >= umbral))
        auc_folds.append(roc_auc_score(y_val, proba_val))
        iteraciones.append(getattr(modelo, "best_iteration_", None))
        importancias.append(_importancia(modelo, columnas))
        print(f"Fold {n}: accuracy = {accuracy_folds[-1]:.4f} | AUC = {auc_folds[-1]:.4f}"
              + (f" | iteraciones = {iteraciones[-1]}" if iteraciones[-1] else ""))
    importancia = None
    if all(i is not None for i in importancias):
        importancia = (pd.concat(importancias, axis=1).mean(axis=1)
                       .sort_values(ascending=False).rename("importancia").to_frame())
        importancia["importancia_pct"] = importancia["importancia"] / importancia["importancia"].sum() * 100
    resultado = {
        "accuracy_folds": accuracy_folds,
        "accuracy_media": float(np.mean(accuracy_folds)),
        "accuracy_desvio": float(np.std(accuracy_folds, ddof=1)),
        "auc_folds": auc_folds,
        "auc": float(roc_auc_score(y, oof)),
        "accuracy_oof": float(accuracy_score(y, oof >= umbral)),
        "oof": oof,
        "proba_test": None,
        "importancia": importancia,
        "iteraciones": iteraciones if any(iteraciones) else None,
        "umbral": umbral,
        "modelos": [],
    }
    print(f"Accuracy CV anidada: {resultado['accuracy_media']:.4f} ± {resultado['accuracy_desvio']:.4f} | "
          f"AUC: {resultado['auc']:.4f}")
    return resultado


# ---------------------------------------------------------------------------------------------------
# CatBoost y combinación
# ---------------------------------------------------------------------------------------------------
def como_texto(datos, columnas):
    """Categóricas como texto para CatBoost, con los NaN como "NA" (notebook del modelo 10)."""
    datos = datos.copy()
    for c in columnas:
        datos[c] = datos[c].astype(object).where(datos[c].notna(), "NA").astype(str)
    return datos


def ajustar_pesos(filas, P, y):
    """Pesos no negativos que suman 1 para combinar las columnas de P, minimizando el log loss en las filas dadas
    (SLSQP desde pesos iguales, con el gradiente analítico del log loss). Notebook del modelo 19."""
    Y = np.asarray(y)
    yy, Q = Y[filas], P[filas]
    objetivo = lambda w: log_loss(yy, np.clip(Q @ w, 1e-7, 1 - 1e-7))

    def gradiente(w):
        p = np.clip(Q @ w, 1e-7, 1 - 1e-7)
        return Q.T @ ((p - yy) / (p * (1 - p))) / len(yy)

    n = P.shape[1]
    r = minimize(objetivo, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                 jac=gradiente, options={"ftol": 1e-10, "maxiter": 200})
    w = np.clip(r.x, 0, None)
    return w / w.sum()


# ---------------------------------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _ids_test_originales():
    """ids de test en el orden del archivo original, para verificar la submission."""
    with open(ARCHIVO_TEST, encoding="utf-8") as f:
        return tuple(json.loads(linea)["id"] for linea in f)


def guardar_submission(ids, proba_test, nombre, umbral=0.5, inicio_id=INICIO_ID):
    """Guarda submissions/{nombre}.csv con las columnas COLUMNA_ID_SUBMISSION y COLUMNA_TARGET_SUBMISSION
    ("new"/"used") y lo verifica.

    ids son los id MLA de test (los que devuelve preparar_datos): no se escriben en el archivo, pero se
    usan para verificar que las filas estén en el orden de test_data.jsonlines. La columna ID es el
    número de fila de test, de inicio_id a inicio_id + 29999.
    """
    col_id, col_target = COLUMNA_ID_SUBMISSION, COLUMNA_TARGET_SUBMISSION
    ids = pd.Series(ids).reset_index(drop=True)
    assert tuple(ids) == _ids_test_originales(), "Las predicciones no están en el mismo orden que test_data.jsonlines"
    submission = pd.DataFrame({
        col_id: np.arange(inicio_id, inicio_id + len(ids)),
        col_target: np.where(np.asarray(proba_test) >= umbral, CLASES[1], CLASES[0]),
    })
    assert len(submission) == FILAS_TEST, f"La submission tiene {len(submission)} filas y debería tener {FILAS_TEST}"
    assert list(submission.columns) == [col_id, col_target], (
        f"Columnas {list(submission.columns)}; deberían ser {[col_id, col_target]}")
    assert submission.notna().all().all(), "La submission tiene nulos"
    assert submission[col_target].isin(CLASES.values()).all(), f"{col_target} tiene valores distintos de new/used"
    assert (submission[col_id].is_unique
            and submission[col_id].tolist() == list(range(inicio_id, inicio_id + FILAS_TEST))), (
        f"{col_id} tiene que ser consecutivo de {inicio_id} a {inicio_id + FILAS_TEST - 1}, sin repetir")

    CARPETA_SUBMISSIONS.mkdir(exist_ok=True)
    archivo = CARPETA_SUBMISSIONS / f"{nombre}.csv"
    submission.to_csv(archivo, index=False)
    print(f"Submission guardada en {archivo.relative_to(CARPETA)}: {len(submission)} filas, "
          f"{(submission[col_target] == CLASES[1]).mean():.1%} used")
    return archivo
