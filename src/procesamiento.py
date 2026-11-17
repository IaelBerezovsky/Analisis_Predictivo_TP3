import json
import unicodedata

import numpy as np
import pandas as pd

# Categorías tomadas del train, fijas para que train y test terminen con las mismas columnas
METODOS_PAGO = [
    "Efectivo", "Transferencia bancaria", "Tarjeta de crédito", "Acordar con el comprador",
    "Giro postal", "MercadoPago", "Visa", "MasterCard", "Visa Electron", "Mastercard Maestro",
    "Contra reembolso", "American Express", "Diners", "Cheque certificado",
]
GRUPOS_ATRIBUTOS = [
    "FIND", "CONFORT", "SECURITY", "DFLT", "ADICIONALES", "SONIDO", "AMBIENTES", "COMODIDADES",
    "COMOYAMEN", "EXTERIOR", "CARACTERISTICAS", "EQUIPAMIENTO", "DDPLANAHO", "RANGCUOTA",
]
TAGS = [
    "dragged_bids_and_visits", "good_quality_thumbnail", "dragged_visits", "free_relist",
    "poor_quality_thumbnail",
]
COLUMNAS_ID_NUMERICAS = ["seller_id", "official_store_id", "catalog_product_id"]
COLUMNAS_CATEGORICAS = [
    "condition", "listing_type_id", "buying_mode", "status", "currency_id", "shipping_mode",
    "seller_address_state_name", "seller_address_state_id",
]
# Categorías de cada columna de COLUMNAS_CATEGORICAS, ordenadas, tal como aparecen en train
CATEGORIAS = {
    "condition": ["new", "used"],
    "listing_type_id": ["bronze", "free", "gold", "gold_premium", "gold_pro", "gold_special", "silver"],
    "buying_mode": ["auction", "buy_it_now", "classified"],
    "status": ["active", "closed", "not_yet_active", "paused"],
    "currency_id": ["ARS", "USD"],
    "shipping_mode": ["custom", "me1", "me2", "not_specified"],
    "seller_address_state_name": [
        "Buenos Aires", "Capital Federal", "Catamarca", "Chaco", "Chubut", "Corrientes", "Córdoba",
        "Entre Ríos", "Formosa", "Jujuy", "La Pampa", "La Rioja", "Mendoza", "Misiones", "Neuquén",
        "Río Negro", "Salta", "San Juan", "San Luis", "Santa Cruz", "Santa Fe", "Santiago del Estero",
        "Tierra del Fuego", "Tucumán",
    ],
    "seller_address_state_id": [
        "AR-A", "AR-B", "AR-C", "AR-D", "AR-E", "AR-F", "AR-G", "AR-H", "AR-J", "AR-K", "AR-L", "AR-M",
        "AR-N", "AR-P", "AR-Q", "AR-R", "AR-S", "AR-T", "AR-U", "AR-V", "AR-W", "AR-X", "AR-Y", "AR-Z",
    ],
}

# Tratamiento de nulos
COLUMNAS_SIN_INFORMACION = [
    "listing_source", "differential_pricing", "subtitle", "coverage_areas", "site_id",
    "international_delivery_mode", "seller_address_country_name", "seller_address_country_id",
    "catalog_product_id", "shipping_dimensions", "shipping_tags", "shipping_methods", "sub_status",
]
COLUMNAS_REDUNDANTES = ["parent_item_id", "seller_address_city_id", "thumbnail", "secure_thumbnail"]
COLUMNAS_A_BOOLEANAS = {
    "video_id": "tiene_video",
    "official_store_id": "es_tienda_oficial",
    "deal_ids": "en_promocion",
    "original_price": "tiene_descuento",
}
COLUMNAS_DIRECCION = ["seller_address_state_name", "seller_address_state_id", "seller_address_city_name"]
DESCONOCIDO = "desconocido"
TEXTOS_SIN_CONTENIDO = ["", ".", "-"]

# Dólar informal de octubre de 2015. Se eligió en lugar del oficial ($9,56) porque durante el cepo
# cambiario las operaciones en dólares entre particulares se hacían a la cotización informal, así que
# refleja mejor el costo real en pesos para el comprador.
TIPO_CAMBIO_USD = 15.90

# Tratamiento de outliers: las stop_time posteriores a este año son una fecha de relleno (2024-11-04 14:55:51)
ANIO_MAXIMO_STOP_TIME = 2020


def leer_jsonlines(path):
    """Lee un archivo JSON Lines y abre los diccionarios con json_normalize.

    Los diccionarios anidados quedan como columnas propias, por ejemplo
    seller_address -> country -> name pasa a ser seller_address_country_name.
    """
    with open(path, encoding="utf-8") as f:
        registros = [json.loads(linea) for linea in f]
    return pd.json_normalize(registros, sep="_")


def cargar_datos(path_train="train_data.jsonlines", path_test="test_data.jsonlines"):
    return leer_jsonlines(path_train), leer_jsonlines(path_test)


def convertir_fechas(df):
    """start_time y stop_time vienen en milisegundos desde 1970."""
    df = df.copy()
    for col in ["start_time", "stop_time"]:
        df[col] = pd.to_datetime(df[col], unit="ms")
    return df


def _como_lista(x):
    return x if isinstance(x, list) else []


def _slug(texto):
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return texto.lower().replace(" ", "_")


def abrir_metodos_pago(df):
    """Una columna booleana por método de pago, más la cantidad de métodos."""
    df = df.copy()
    metodos = df["non_mercado_pago_payment_methods"].map(
        lambda l: {e["description"] for e in _como_lista(l)}
    )
    for metodo in METODOS_PAGO:
        df[f"pago_{_slug(metodo)}"] = metodos.map(lambda s: metodo in s)
    df["pago_cantidad_metodos"] = metodos.map(len)
    return df.drop(columns="non_mercado_pago_payment_methods")


def abrir_variaciones(df):
    """Resume las variaciones (talles/colores) de cada publicación."""
    df = df.copy()
    variaciones = df["variations"].map(_como_lista)

    def valores(lista, nombre):
        return {a["value_name"] for v in lista for a in v["attribute_combinations"] if a["name"] == nombre}

    # El precio de las variaciones no se guarda: siempre coincide con price
    df["variaciones_cantidad"] = variaciones.map(len)
    df["variaciones_vendidas"] = variaciones.map(lambda l: sum(v["sold_quantity"] for v in l))
    df["variaciones_disponibles"] = variaciones.map(lambda l: sum(v["available_quantity"] for v in l))
    df["variaciones_colores"] = variaciones.map(lambda l: len(valores(l, "Color Primario")))
    df["variaciones_talles"] = variaciones.map(lambda l: len(valores(l, "Talle")))
    return df.drop(columns="variations")


def abrir_atributos(df):
    """Cantidad de atributos y una columna booleana por grupo de atributos."""
    df = df.copy()
    atributos = df["attributes"].map(_como_lista)
    grupos = atributos.map(lambda l: {e["attribute_group_id"] for e in l})
    df["atributos_cantidad"] = atributos.map(len)
    for grupo in GRUPOS_ATRIBUTOS:
        df[f"atributos_grupo_{grupo.lower()}"] = grupos.map(lambda s: grupo in s)
    return df.drop(columns="attributes")


def _pixeles(foto):
    # 180 fotos vienen con max_size vacío; en esos casos se usa size
    ancho, alto = (foto["max_size"] or foto["size"]).split("x")
    return int(ancho) * int(alto)


def abrir_fotos(df):
    """Cantidad de fotos y resolución (en píxeles) de la foto más grande."""
    df = df.copy()
    fotos = df["pictures"].map(_como_lista)
    df["fotos_cantidad"] = fotos.map(len)
    df["fotos_max_pixeles"] = fotos.map(lambda l: max((_pixeles(p) for p in l), default=np.nan))
    return df.drop(columns="pictures")


def abrir_envio_gratis(df):
    """shipping_free_methods solo trae un id y la regla 'country': alcanza con la cantidad."""
    df = df.copy()
    df["shipping_free_methods_cantidad"] = df["shipping_free_methods"].map(lambda l: len(_como_lista(l)))
    return df.drop(columns="shipping_free_methods")


def convertir_fechas_texto(df):
    """date_created y last_updated vienen como texto ISO en UTC.

    Se les saca la zona horaria para que sean comparables con start_time y stop_time.
    """
    df = df.copy()
    for col in ["date_created", "last_updated"]:
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
    return df


def vacios_a_nulos(df):
    """Los textos vacíos ('') son datos faltantes."""
    df = df.copy()
    cols_texto = df.select_dtypes(include="str").columns
    df[cols_texto] = df[cols_texto].replace("", np.nan)
    return df


def ids_a_texto(df):
    """Los IDs numéricos son identificadores, no cantidades.

    Se pasan a texto sin '.0' y los faltantes quedan como nulos (no como el texto '<NA>').
    """
    df = df.copy()
    for col in COLUMNAS_ID_NUMERICAS:
        df[col] = df[col].map(lambda x: str(int(x)), na_action="ignore")
    return df


def abrir_listas_simples(df):
    """Convierte las listas de valores simples en columnas escalares.

    sub_status, deal_ids y shipping_tags tienen como máximo un elemento, así que
    pasan a ser texto. tags tiene hasta dos, así que se abre en una columna
    booleana por tag. descriptions solo trae un id, así que queda si la
    publicación tiene descripción o no.
    """
    df = df.copy()
    for col in ["sub_status", "deal_ids", "shipping_tags"]:
        # Sin astype("str"), que en algunas versiones de pandas convierte los nulos en el texto 'nan'
        df[col] = df[col].map(lambda l: _como_lista(l)[0] if _como_lista(l) else np.nan)
    tags = df["tags"].map(lambda l: set(_como_lista(l)))
    for tag in TAGS:
        df[f"tag_{tag}"] = tags.map(lambda s: tag in s)
    df["tiene_descripcion"] = df["descriptions"].map(lambda l: len(_como_lista(l)) > 0)
    return df.drop(columns=["tags", "descriptions"])


def a_categoricas(df):
    """Variables de texto con pocos valores posibles (condition no está en test).

    Las categorías son siempre las de train (CATEGORIAS), así train y test tienen
    las mismas categorías y los mismos códigos. Un valor que no esté en train queda como NaN.
    """
    df = df.copy()
    for col in COLUMNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype(pd.CategoricalDtype(CATEGORIAS[col]))
    return df


def eliminar_sin_informacion(df):
    """Columnas vacías, constantes o con más de 99,9% de nulos (ver análisis de nulos)."""
    return df.drop(columns=COLUMNAS_SIN_INFORMACION)


def eliminar_redundantes(df):
    """parent_item_id lo indica tag_dragged_bids_and_visits; seller_address_city_id, la ciudad en
    seller_address_city_name; thumbnail y secure_thumbnail son URLs y fotos_cantidad indica si hay fotos."""
    return df.drop(columns=COLUMNAS_REDUNDANTES)


def nulos_a_booleanas(df):
    """Variables casi vacías cuya señal es tener o no valor: se reemplazan por una booleana."""
    df = df.copy()
    for col, nueva in COLUMNAS_A_BOOLEANAS.items():
        df[nueva] = df[col].notna()
    return df.drop(columns=list(COLUMNAS_A_BOOLEANAS))


def tratar_warranty(df):
    """Textos sin contenido -> nulo; garantia_cargada indica si hay texto; los nulos pasan a 'desconocido'."""
    df = df.copy()
    sin_contenido = df["warranty"].str.strip().isin(TEXTOS_SIN_CONTENIDO)
    df["warranty"] = df["warranty"].mask(sin_contenido)
    df["garantia_cargada"] = df["warranty"].notna()
    df["warranty"] = df["warranty"].fillna(DESCONOCIDO)
    return df


def tratar_direccion(df):
    """Ciudad '-' o '.' y nulos de la dirección -> 'desconocido' (también como categoría)."""
    df = df.copy()
    ciudad = df["seller_address_city_name"]
    df["seller_address_city_name"] = ciudad.mask(ciudad.str.strip().isin(["-", "."]), DESCONOCIDO)
    for col in COLUMNAS_DIRECCION:
        if isinstance(df[col].dtype, pd.CategoricalDtype) and DESCONOCIDO not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories(DESCONOCIDO)
        df[col] = df[col].fillna(DESCONOCIDO)
    return df


def fotos_sin_pixeles(df):
    """fotos_max_pixeles es nulo solo en publicaciones sin fotos (fotos_cantidad = 0)."""
    df = df.copy()
    df["fotos_max_pixeles"] = df["fotos_max_pixeles"].fillna(0)
    return df


def es_precio_relleno(precio):
    """Precios <= 1, formados por un único dígito repetido 5 o más veces (11111, 8888888...) o por
    5 o más dígitos consecutivos, ascendentes o descendentes (12345, 123456789, 54321...)."""
    texto = precio.map(lambda x: f"{x:.0f}", na_action="ignore")
    repetido = texto.str.fullmatch(r"(\d)\1{4,}")
    consecutivo = (texto.str.len() >= 5) & texto.map(
        lambda t: t in "1234567890" or t in "9876543210", na_action="ignore"
    )
    return (precio <= 1) | repetido.fillna(False).astype(bool) | consecutivo.fillna(False).astype(bool)


def precios_relleno_a_nulos(df):
    """Los precios de relleno no son el precio real: quedan como nulos (no se imputan)."""
    df = df.copy()
    for col in ["price", "base_price"]:
        df[col] = df[col].mask(es_precio_relleno(df[col]))
    return df


def precios_usd_a_pesos(df):
    """price y base_price de las publicaciones en USD pasan a pesos (TIPO_CAMBIO_USD).

    Va después de precios_relleno_a_nulos, que se evalúa sobre el precio original. Los NaN siguen
    siendo NaN y currency_id no se modifica.
    """
    df = df.copy()
    en_usd = df["currency_id"] == "USD"
    df.loc[en_usd, ["price", "base_price"]] *= TIPO_CAMBIO_USD
    return df


def eliminar_base_price(df):
    """base_price es igual a price salvo en 4 filas y genera filas donde solo uno de los dos es NaN."""
    return df.drop(columns="base_price")


def stop_time_relleno_a_nulos(df):
    """Las stop_time posteriores a 2020 son una fecha de relleno (2024-11-04 14:55:51): pasan a NaT."""
    df = df.copy()
    df["stop_time"] = df["stop_time"].mask(df["stop_time"].dt.year > ANIO_MAXIMO_STOP_TIME)
    return df


def procesar(df):
    """Aplica en orden todas las transformaciones definidas en el EDA."""
    df = convertir_fechas(df)
    df = abrir_metodos_pago(df)
    df = abrir_variaciones(df)
    df = abrir_atributos(df)
    df = abrir_fotos(df)
    df = abrir_envio_gratis(df)
    df = convertir_fechas_texto(df)
    df = vacios_a_nulos(df)
    df = ids_a_texto(df)
    df = abrir_listas_simples(df)
    df = a_categoricas(df)
    df = eliminar_sin_informacion(df)
    df = eliminar_redundantes(df)
    df = nulos_a_booleanas(df)
    df = tratar_warranty(df)
    df = tratar_direccion(df)
    df = fotos_sin_pixeles(df)
    df = precios_relleno_a_nulos(df)
    df = precios_usd_a_pesos(df)  # después de precios_relleno_a_nulos: la regla usa el precio original
    df = eliminar_base_price(df)
    df = stop_time_relleno_a_nulos(df)
    return df


if __name__ == "__main__":
    df_train, df_test = cargar_datos()
    df_train = procesar(df_train)
    df_test = procesar(df_test)
    print(df_train.shape, df_test.shape)
