"""Feature engineering del modelo final (copiado sin cambios de features.py del proyecto; solo las funciones que usa
el modelo 19).

Las funciones agregar_features_* reciben un DataFrame ya procesado con procesar() y devuelven una copia con columnas
nuevas. Se aplican por separado a train y a test, así que no usan información de otras filas ni el target.
leer_fotos_atributos() y leer_json_crudo() leen el JSON original, porque necesitan datos que procesar() no conserva.
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------
# Normalización de texto (análisis de palabras del EDA)
# ---------------------------------------------------------------------------------------------------
# Caracteres con problemas de codificación vistos en los datos: "Ituzaingã¿" y "Morã¿n" (ó), "Mejã\x8da" (í),
# guiones suaves invisibles (\xad) y espacios duros (\xa0)
REPARACIONES_CODIFICACION = {"ã¿": "ó", "ã\x8d": "í", "\xad": "", "\xa0": " "}


def normalizar_texto(texto):
    """Minúsculas, sin tildes, con los problemas de codificación corregidos y solo letras, números y espacios."""
    if not isinstance(texto, str):
        return ""
    if "Ã" in texto or "Â" in texto:  # texto UTF-8 leído como latin-1 ("Ã³" en lugar de "ó")
        try:
            texto = texto.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    for malo, bueno in REPARACIONES_CODIFICACION.items():
        texto = texto.replace(malo, bueno)
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", texto).strip()


# Patrones de warranty, sobre el texto normalizado
PATRONES_WARRANTY = {
    "sin garantía": r"\bsin garantia|\bno (?:tiene|posee|ofrece|ofrezco|doy|incluye|hay) garantia|^no$",
    "menciona meses": r"\bmes(?:es)?\b",
    "menciona años": r"\banos?\b",
    "menciona días": r"\bdias?\b",
    "de fábrica": r"\bfabrica(?:nte)?\b",
    "oficial": r"\boficial(?:es)?\b",
    "del vendedor": r"\bvendedor\b",
    "'si' o 'sí' solo": r"^si$",
}

# ---------------------------------------------------------------------------------------------------
# Variables de texto de title y warranty (modelo 02)
# ---------------------------------------------------------------------------------------------------
WARRANTY_NO_CARGADA = "desconocido"

# Booleanas del título: palabras o frases completas sobre el título normalizado
PALABRAS_TITULO = {
    "titulo_usado": ["usado", "usada", "usados", "usadas"],
    "titulo_estado": ["estado", "estados"],
    "titulo_impecable": ["impecable", "impecables"],
    "titulo_permuto": ["permuto"],
    "titulo_poco_uso": ["poco uso"],
    "titulo_vendo": ["vendo"],
    "titulo_muy": ["muy"],
    "titulo_nuevo": ["nuevo", "nuevos"],
    "titulo_nueva": ["nueva", "nuevas"],
    "titulo_a_estrenar": ["a estrenar"],
    "titulo_cerrado": ["cerrado", "cerrada", "cerrados", "cerradas", "sellado", "sellada", "sellados", "selladas"],
    "titulo_digital": ["digital", "digitales"],
    "titulo_stock": ["stock"],
    "titulo_mint": ["mint"],
    "titulo_neumatico": ["neumatico", "neumaticos"],
}

# Booleanas de warranty: patrones del EDA y palabras o frases completas sobre warranty normalizada
PATRONES_GARANTIA = {
    "garantia_sin": PATRONES_WARRANTY["sin garantía"],
    "garantia_meses": PATRONES_WARRANTY["menciona meses"],
    "garantia_anos": PATRONES_WARRANTY["menciona años"],
    "garantia_dias": PATRONES_WARRANTY["menciona días"],
    "garantia_fabrica": PATRONES_WARRANTY["de fábrica"],
    "garantia_oficial": PATRONES_WARRANTY["oficial"],
}
PALABRAS_GARANTIA = {
    "garantia_nuevo": ["nuevo", "nueva", "nuevos", "nuevas"],
    "garantia_formal": [
        "fabricacion", "falla", "fallas", "factura", "escrita", "escrito", "importador", "distribuidor",
        "distribuidores", "directo", "directa", "vencimiento", "cubre", "servicio", "mal uso", "garantia total",
        "cuenta con", "cuentan con", "marca", "marcas",
    ],
}

COLUMNAS_BOOLEANAS_TITULO = list(PALABRAS_TITULO)
COLUMNAS_BOOLEANAS_WARRANTY = list(PATRONES_GARANTIA) + list(PALABRAS_GARANTIA)
COLUMNAS_METRICAS_TITULO = ["titulo_largo", "titulo_palabras", "titulo_pct_mayusculas"]
COLUMNAS_TEXTO = COLUMNAS_BOOLEANAS_TITULO + COLUMNAS_BOOLEANAS_WARRANTY + COLUMNAS_METRICAS_TITULO


def regex_palabras(palabras):
    """Regex que encuentra cualquiera de las palabras o frases como palabra completa."""
    return r"\b(?:" + "|".join(re.escape(p) for p in palabras) + r")\b"


def _pct_mayusculas(texto):
    letras = [c for c in texto if c.isalpha()]
    return sum(c.isupper() for c in letras) / len(letras) if letras else 0.0


def agregar_features_texto(df):
    """Booleanas de palabras de title y warranty y métricas simples del título (ver COLUMNAS_TEXTO).

    Las booleanas de warranty valen False cuando la garantía no está cargada ("desconocido").
    """
    df = df.copy()
    titulo = df["title"].fillna("")
    titulo_normalizado = titulo.map(normalizar_texto)
    for columna, palabras in PALABRAS_TITULO.items():
        df[columna] = titulo_normalizado.str.contains(regex_palabras(palabras), regex=True)

    cargada = df["warranty"] != WARRANTY_NO_CARGADA
    warranty_normalizada = df["warranty"].map(normalizar_texto)
    for columna, regex in PATRONES_GARANTIA.items():
        df[columna] = warranty_normalizada.str.contains(regex, regex=True) & cargada
    for columna, palabras in PALABRAS_GARANTIA.items():
        df[columna] = warranty_normalizada.str.contains(regex_palabras(palabras), regex=True) & cargada

    df["titulo_largo"] = titulo.str.len()
    df["titulo_palabras"] = titulo_normalizado.str.split().str.len()
    df["titulo_pct_mayusculas"] = titulo.map(_pct_mayusculas)
    return df


# ---------------------------------------------------------------------------------------------------
# Variables de vendedor y categoría sin target (modelo 03)
# ---------------------------------------------------------------------------------------------------
COLUMNAS_FRECUENCIA = ["frecuencia_vendedor", "frecuencia_categoria", "frecuencia_ciudad"]
COLUMNAS_PERFIL_VENDEDOR = [
    "vendedor_mediana_stock", "vendedor_pct_free", "vendedor_pct_gold_premium", "vendedor_mediana_precio",
    "vendedor_media_metodos_pago", "vendedor_pct_garantia", "vendedor_pct_me2", "vendedor_categorias",
    "vendedor_media_fotos",
]
COLUMNAS_PRECIO_RELATIVO = ["precio_relativo_categoria"]
COLUMNAS_SIN_TARGET = COLUMNAS_FRECUENCIA + COLUMNAS_PERFIL_VENDEDOR + COLUMNAS_PRECIO_RELATIVO
MINIMO_PUBLICACIONES_CATEGORIA = 5


def variables_vendedor_categoria(df_train, df_test):
    """Frecuencias, perfil del vendedor y precio relativo a la categoría (ver COLUMNAS_SIN_TARGET).

    No usan condition, así que las frecuencias y el perfil del vendedor se calculan sobre train + test
    juntos. La mediana de precio por categoría sale solo de train, y solo para categorías con al menos
    MINIMO_PUBLICACIONES_CATEGORIA publicaciones en train (si no, el precio relativo queda NaN).

    Devuelve dos DataFrames (train y test) con las columnas nuevas y el mismo índice que los de entrada.
    """
    todo = pd.concat([df_train, df_test], keys=["train", "test"])
    nuevas = pd.DataFrame(index=todo.index)

    for columna, origen in zip(COLUMNAS_FRECUENCIA, ["seller_id", "category_id", "seller_address_city_name"]):
        nuevas[columna] = todo.groupby(origen)[origen].transform("size")

    vendedor = todo.assign(
        es_free=todo["listing_type_id"] == "free",
        es_gold_premium=todo["listing_type_id"].isin(["gold_special", "gold_premium"]),
        es_me2=todo["shipping_mode"] == "me2",
    ).groupby("seller_id")
    nuevas["vendedor_mediana_stock"] = vendedor["initial_quantity"].transform("median")
    nuevas["vendedor_pct_free"] = vendedor["es_free"].transform("mean")
    nuevas["vendedor_pct_gold_premium"] = vendedor["es_gold_premium"].transform("mean")
    nuevas["vendedor_mediana_precio"] = vendedor["price"].transform("median")
    nuevas["vendedor_media_metodos_pago"] = vendedor["pago_cantidad_metodos"].transform("mean")
    nuevas["vendedor_pct_garantia"] = vendedor["garantia_cargada"].transform("mean")
    nuevas["vendedor_pct_me2"] = vendedor["es_me2"].transform("mean")
    nuevas["vendedor_categorias"] = vendedor["category_id"].transform("nunique")
    nuevas["vendedor_media_fotos"] = vendedor["fotos_cantidad"].transform("mean")

    por_categoria = df_train.groupby("category_id")["price"].agg(["median", "size"])
    mediana = por_categoria.loc[por_categoria["size"] >= MINIMO_PUBLICACIONES_CATEGORIA, "median"]
    nuevas["precio_relativo_categoria"] = todo["price"] / todo["category_id"].map(mediana)

    return nuevas.loc["train"].set_index(df_train.index), nuevas.loc["test"].set_index(df_test.index)


# ---------------------------------------------------------------------------------------------------
# Fechas (modelo 05)
# ---------------------------------------------------------------------------------------------------
COLUMNAS_FECHAS = ["dias_hasta_ultima_actualizacion", "dia_semana", "hora_creacion"]


def agregar_features_fechas(df):
    """Días entre date_created y last_updated, día de la semana (0 = lunes) y hora de date_created."""
    df = df.copy()
    df["dias_hasta_ultima_actualizacion"] = (df["last_updated"] - df["date_created"]).dt.total_seconds() / 86400
    df["dia_semana"] = df["date_created"].dt.dayofweek
    df["hora_creacion"] = df["date_created"].dt.hour
    return df


# ---------------------------------------------------------------------------------------------------
# Fotos y atributos del JSON original (modelo 07)
# ---------------------------------------------------------------------------------------------------
def leer_fotos_atributos(ruta_jsonlines):
    """Lee del JSON original, por publicación y en el orden del archivo: id, seller_id, category_id, date_created
    (sin zona horaria), condition (NaN en test), pictures (lista de dicts con id, url, size y max_size) y
    attributes (lista de dicts con id, name, value_id, value_name y attribute_group_id)."""
    campos_foto = ["id", "url", "size", "max_size"]
    campos_atributo = ["id", "name", "value_id", "value_name", "attribute_group_id"]
    filas = []
    with open(ruta_jsonlines, encoding="utf-8") as f:
        for linea in f:
            registro = json.loads(linea)
            filas.append({
                "id": registro["id"],
                "seller_id": registro["seller_id"],
                "category_id": registro["category_id"],
                "date_created": registro["date_created"],
                "condition": registro.get("condition"),
                "pictures": [{c: p.get(c) for c in campos_foto} for p in registro.get("pictures") or []],
                "attributes": [{c: a.get(c) for c in campos_atributo} for a in registro.get("attributes") or []],
            })
    datos = pd.DataFrame(filas)
    datos["date_created"] = pd.to_datetime(datos["date_created"], utc=True).dt.tz_localize(None)
    return datos


# Id de foto: "{número}-MLA{número}_{sufijo}". Cuando el sufijo tiene 6 dígitos es MMAAAA, el mes en que se subió
# la foto; el formato viejo trae un sufijo de 4 dígitos sin fecha (ejemplo: "4005-MLA3911152707_4729").
REGEX_ID_FOTO = re.compile(r"^(\d+)-MLA(\d+)_(\d+)$")
COLUMNAS_FOTOS = ["fotos_pct_sin_fecha", "fotos_meses_desde_mas_vieja", "fotos_meses_desde_mas_nueva", "foto_id_numero"]


def mes_id_foto(id_foto):
    """Mes de subida de la foto como año * 12 + (mes - 1), o None si el id no trae fecha."""
    partes = REGEX_ID_FOTO.match(id_foto or "")
    if partes is None or len(partes.group(3)) != 6:
        return None
    mes, anio = int(partes.group(3)[:2]), int(partes.group(3)[2:])
    return anio * 12 + mes - 1 if 1 <= mes <= 12 and 2000 <= anio <= 2016 else None


def variables_fotos(datos):
    """Variables del id de las fotos, sin target (datos de leer_fotos_atributos). NaN si la publicación no tiene fotos.

    - fotos_pct_sin_fecha: % de las fotos con id del formato viejo, sin fecha de subida.
    - fotos_meses_desde_mas_vieja / _mas_nueva: meses entre la subida de la foto más vieja (o más nueva) y la
      creación de la publicación. NaN si ninguna foto trae fecha.
    - foto_id_numero: el número inicial del id de la primera foto.
    """
    filas = []
    for fotos, creada in zip(datos["pictures"], datos["date_created"]):
        meses = [m for m in (mes_id_foto(p["id"]) for p in fotos) if m is not None]
        primera = REGEX_ID_FOTO.match(fotos[0]["id"] or "") if fotos else None
        mes_publicacion = creada.year * 12 + creada.month - 1
        filas.append({
            "fotos_pct_sin_fecha": (1 - len(meses) / len(fotos)) * 100 if fotos else np.nan,
            "fotos_meses_desde_mas_vieja": mes_publicacion - min(meses) if meses else np.nan,
            "fotos_meses_desde_mas_nueva": mes_publicacion - max(meses) if meses else np.nan,
            "foto_id_numero": float(primera.group(1)) if primera else np.nan,
        })
    return pd.DataFrame(filas, index=datos.index)


COLUMNAS_ATRIBUTOS_NUMERICAS = ["km_vehiculo", "anio_vehiculo", "antiguedad_inmueble"]
COLUMNAS_ATRIBUTOS_CATEGORICAS = ["unico_duenio", "estado_inmueble", "genero", "temporada"]
COLUMNAS_ATRIBUTOS = COLUMNAS_ATRIBUTOS_NUMERICAS + COLUMNAS_ATRIBUTOS_CATEGORICAS + ["tiene_numero_pieza"]
# Atributo de origen de cada variable: sufijo del id (los vehículos e inmuebles los nombran por categoría, por
# ejemplo MLA1744-KMTS y MLA1763-KMTS)
SUFIJOS_ATRIBUTOS = {"km_vehiculo": "KMTS", "anio_vehiculo": "YEAR", "antiguedad_inmueble": "ANTIG",
                     "unico_duenio": "OWNER", "estado_inmueble": "ESTADO", "genero": "GENDER", "temporada": "Season"}


def _valor_atributo(atributos, sufijo):
    """value_name del primer atributo cuyo id es `sufijo` o termina en "-{sufijo}"; None si no lo tiene."""
    for a in atributos:
        if a["id"] == sufijo or a["id"].endswith(f"-{sufijo}"):
            return (a["value_name"] or "").strip()
    return None


def _numero(texto):
    digitos = re.sub(r"\D", "", texto) if isinstance(texto, str) else ""
    return float(digitos) if digitos else np.nan


def _antiguedad(texto):
    """Años de antigüedad: "A estrenar" = 0, "Hasta 5 años" o "Max 5 años" = 5, "Más de 40 años" = 41."""
    if not isinstance(texto, str):
        return np.nan
    if "estrenar" in texto.lower():
        return 0.0
    anios = _numero(texto)
    return anios + 1 if texto.lower().startswith("m\u00e1s de") else anios


def variables_atributos(datos_train, datos_test):
    """Valores de atributos como variables (datos de leer_fotos_atributos), NaN donde la publicación no tiene el
    atributo.

    - km_vehiculo y anio_vehiculo (vehículos; años fuera de 1900-2016 pasan a NaN) y antiguedad_inmueble
      (inmuebles, en años).
    - unico_duenio, estado_inmueble, genero y temporada: categóricas con las categorías de train; un atributo
      cargado sin valor queda como "sin dato".
    - tiene_numero_pieza: tiene el atributo PART_NUMBER (repuestos, casi siempre new).
    Devuelve (DataFrame de train, DataFrame de test).
    """
    def calcular(datos):
        valores = {col: datos["attributes"].map(lambda l, s=sufijo: _valor_atributo(l, s))
                   for col, sufijo in SUFIJOS_ATRIBUTOS.items()}
        anio = valores["anio_vehiculo"].map(_numero)
        variables = pd.DataFrame({
            "km_vehiculo": valores["km_vehiculo"].map(_numero),
            "anio_vehiculo": anio.where(anio.between(1900, 2016)),
            "antiguedad_inmueble": valores["antiguedad_inmueble"].map(_antiguedad),
        }, index=datos.index)
        for col in COLUMNAS_ATRIBUTOS_CATEGORICAS:
            variables[col] = valores[col].map(lambda v: (v or "sin dato") if isinstance(v, str) else None)
        variables["tiene_numero_pieza"] = datos["attributes"].map(
            lambda l: any(a["id"] in ("PART_NUMBER", "MLA-PART_NUMBER") for a in l)).astype(int)
        return variables

    train, test = calcular(datos_train), calcular(datos_test)
    for col in COLUMNAS_ATRIBUTOS_CATEGORICAS:
        tipo = pd.CategoricalDtype(sorted(train[col].dropna().unique()))
        train[col], test[col] = train[col].astype(tipo), test[col].astype(tipo)
    return train, test


# ---------------------------------------------------------------------------------------------------
# Resto del JSON original (modelo 08)
# ---------------------------------------------------------------------------------------------------
def leer_json_crudo(ruta_jsonlines):
    """Todas las claves del JSON original sin transformar, una fila por publicación en el orden del archivo."""
    with open(ruta_jsonlines, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(linea) for linea in f])


COLUMNAS_DURACION = ["duracion_dias", "duracion_segundos_extra", "segundos_inicio_desde_creacion"]


def agregar_features_duracion(df):
    """Duración de la publicación y desfase del inicio (start_time y stop_time no entran en el modelo; procesar()
    deja NaN los stop_time de relleno).

    - duracion_dias: stop_time - start_time, en días.
    - duracion_segundos_extra: segundos por encima del día entero más cercano. El 94% dura exactamente 60 días;
      las que duran 60 días y unos segundos son casi todas used (modelo 08). Se separa porque LightGBM agrupa los
      valores en bins y no distinguiría 60 de 60,0001 días.
    - segundos_inicio_desde_creacion: start_time - date_created.
    """
    df = df.copy()
    segundos = (df["stop_time"] - df["start_time"]).dt.total_seconds()
    df["duracion_dias"] = segundos / 86400
    df["duracion_segundos_extra"] = segundos - (segundos / 86400).round() * 86400
    df["segundos_inicio_desde_creacion"] = (df["start_time"] - df["date_created"]).dt.total_seconds()
    return df


# ---------------------------------------------------------------------------------------------------
# Valores de los campos del JSON y patrones del análisis de errores (modelo 12)
# ---------------------------------------------------------------------------------------------------
def valores_campos_json(crudo):
    """Valores de cada campo del JSON original por fila, para el barrido de residuos del modelo 12.

    Recibe lo que devuelve leer_json_crudo(). Devuelve un dict fuente -> Series (con el índice de `crudo`) de
    conjuntos de valores: palabras y pares de palabras del título normalizado, palabras de la garantía, categoría,
    atributos (id e id=valor), tags, medios de pago, envío, ciudad, provincia, otros campos crudos (variaciones,
    deal_ids, precio original, relist, status, sub_status, video, MercadoPago, descripciones), terminación del
    precio y precio redondo (múltiplo de 100) dentro de cada categoría.
    """
    titulo = crudo["title"].map(lambda t: normalizar_texto(t).split() if isinstance(t, str) else [])
    precio = pd.to_numeric(crudo["price"], errors="coerce").to_numpy()
    envio = crudo["shipping"].map(lambda s: s if isinstance(s, dict) else {})

    def terminacion(x):
        if x != x or x != int(x):
            return {"con centavos"}
        return {f"termina en {int(x) % 1000:03d}", f"termina en {int(x) % 10}"}

    otros = [{f"variaciones={min(len(v or []), 3)}", f"deal_ids={len(d or []) > 0}", f"precio_original={pd.notna(o)}",
              f"relist={a}", f"status={s}", f"sub_status={','.join(ss or [])}", f"video={pd.notna(vi)}",
              f"mercadopago={mp}", f"descripciones={len(de or [])}"}
             for v, d, o, a, s, ss, vi, mp, de in zip(
                 crudo["variations"], crudo["deal_ids"], crudo["original_price"], crudo["automatic_relist"],
                 crudo["status"], crudo["sub_status"], crudo["video_id"], crudo["accepts_mercadopago"],
                 crudo["descriptions"])]
    return {
        "título (palabra)": titulo.map(set),
        "título (par de palabras)": titulo.map(lambda t: {" ".join(z) for z in zip(t, t[1:])}),
        "garantía (palabra)": crudo["warranty"].map(
            lambda t: set(normalizar_texto(t).split()) if isinstance(t, str) else {"<sin garantía>"}),
        "categoría": crudo["category_id"].map(lambda c: {c}),
        "atributo (id)": crudo["attributes"].map(lambda v: {a.get("id") for a in (v or [])} or {"<sin atributos>"}),
        "atributo (id=valor)": crudo["attributes"].map(lambda v: {f"{a.get('id')}={a.get('value_name')}" for a in (v or [])}),
        "tag": crudo["tags"].map(lambda v: set(v or []) or {"<sin tags>"}),
        "medio de pago": crudo["non_mercado_pago_payment_methods"].map(
            lambda v: {d.get("id") for d in (v or [])} or {"<ninguno>"}),
        "envío": envio.map(lambda s: {f"{k}={s.get(k)}" for k in ("mode", "free_shipping", "local_pick_up")}
                           | {f"métodos={len(s.get('methods') or [])}", f"tags={','.join(sorted(s.get('tags') or []))}"}),
        "ciudad": crudo["seller_address"].map(lambda a: {((a or {}).get("city") or {}).get("name")}),
        "provincia": crudo["seller_address"].map(lambda a: {((a or {}).get("state") or {}).get("name")}),
        "otros campos": pd.Series(otros, index=crudo.index),
        "precio": pd.Series([terminacion(x) for x in precio], index=crudo.index),
        "categoría y precio redondo": pd.Series([{f"{c} | múltiplo de 100={x == x and x % 100 == 0}"}
                                                 for c, x in zip(crudo["category_id"], precio)], index=crudo.index),
    }


def variables_patrones(valores, patrones):
    """Booleanas patron_1..patron_n: si la fila tiene cada (fuente, valor) de `patrones`, con `valores` de
    valores_campos_json(). Los patrones salen del barrido de residuos del modelo 12."""
    indice = next(iter(valores.values())).index
    return pd.DataFrame({f"patron_{n}": valores[fuente].map(lambda s, v=valor: v in s).astype(int).to_numpy()
                         for n, (fuente, valor) in enumerate(patrones, start=1)}, index=indice)


# ---------------------------------------------------------------------------------------------------
# Modelo 19: categoría descrita por sus títulos
# ---------------------------------------------------------------------------------------------------
N_COMPONENTES_CATEGORIA = 16
K_CATEGORIAS_VECINAS = 10
SEMILLA_SVD = 42
COLUMNAS_CATEGORIA_SVD = [f"cat_svd_{i}" for i in range(1, N_COMPONENTES_CATEGORIA + 1)]


def categorias_por_titulos(df_train, df_test, k=K_CATEGORIAS_VECINAS, n_componentes=N_COMPONENTES_CATEGORIA, bloque=1000):
    """Describe cada category_id con sus títulos (train + test, sin target).

    Un documento por categoría con los títulos normalizados de todas sus publicaciones; TF-IDF de palabras
    (min_df=2, sublinear_tf) de esos documentos. Devuelve:
    - svd: DataFrame indexado por category_id con cat_svd_1..n (TruncatedSVD, random_state=42);
    - vecinas: DataFrame indexado por category_id con "vecinas" (array de las k categorías más parecidas por
      similitud coseno, sin la propia, elegidas entre las que tienen publicaciones en train) y "similitudes";
    - documentos: Serie category_id -> documento (para inspeccionar).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    categoria = pd.concat([df_train["category_id"], df_test["category_id"]]).astype(str).to_numpy()
    titulos = pd.concat([df_train["title"], df_test["title"]]).map(normalizar_texto).to_numpy()
    documentos = pd.Series(titulos).groupby(categoria).agg(" ".join)
    tfidf = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(documentos.to_numpy())
    svd = TruncatedSVD(n_componentes, random_state=SEMILLA_SVD).fit_transform(tfidf)
    svd = pd.DataFrame(svd, index=documentos.index, columns=[f"cat_svd_{i}" for i in range(1, n_componentes + 1)])

    en_train = np.isin(documentos.index, df_train["category_id"].astype(str).unique())
    candidatas = np.flatnonzero(en_train)
    matriz_candidatas = tfidf[candidatas].T.tocsc()
    vecinas, similitudes = [], []
    for inicio in range(0, tfidf.shape[0], bloque):
        sim = (tfidf[inicio:inicio + bloque] @ matriz_candidatas).toarray()
        propias = np.arange(inicio, min(inicio + bloque, tfidf.shape[0]))
        sim[candidatas[None, :] == propias[:, None]] = -1  # excluir la propia categoría
        orden = np.argpartition(-sim, k, axis=1)[:, :k]
        sim_k = np.take_along_axis(sim, orden, axis=1)
        orden2 = np.argsort(-sim_k, axis=1)
        vecinas.append(candidatas[np.take_along_axis(orden, orden2, axis=1)])
        similitudes.append(np.take_along_axis(sim_k, orden2, axis=1))
    vecinas, similitudes = np.vstack(vecinas), np.clip(np.vstack(similitudes), 0, None)
    tabla = pd.DataFrame({"vecinas": list(documentos.index.to_numpy()[vecinas]), "similitudes": list(similitudes)},
                         index=documentos.index)
    return svd, tabla, documentos
