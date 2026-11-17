# Cómo correr el fine-tuning de títulos en Kaggle

El notebook `kaggle/finetune_titulos.ipynb` entrena un modelo de lenguaje en español (BETO) sobre los títulos, con los mismos 5 folds del proyecto. Necesita una GPU, que esta computadora no tiene; Kaggle la da gratis. Con una GPU T4 tarda **~35-50 minutos**.

Vas a necesitar dos archivos de la carpeta `kaggle/` del proyecto:
- `folds.csv`: qué fold le toca a cada fila de train (lo mismo que `obtener_folds(y)`);
- `finetune_titulos.ipynb`: el notebook.

---

## 1. Subir `folds.csv` como dataset privado

1. Entrá a [kaggle.com](https://www.kaggle.com) con tu cuenta.
2. En el menú de la izquierda, **Datasets** → botón **+ New Dataset**.
3. Arrastrá `kaggle/folds.csv` (o hacé clic en *Browse Files* y elegilo).
4. Poné un título, por ejemplo `folds-tp3`.
5. Dejá la visibilidad en **Private** y tocá **Create**.

Tarda unos segundos en procesarse.

## 2. Crear el notebook desde la competencia

1. Entrá a la página de la competencia (la misma donde subís las submissions).
2. Pestaña **Code** → botón **+ New Notebook**. Se abre un notebook vacío que ya tiene los datos de la competencia como entrada.
3. Menú **File** → **Import Notebook** → elegí `kaggle/finetune_titulos.ipynb` → **Import**. El contenido reemplaza al notebook vacío.

## 3. Agregar los datos de entrada

En el panel de la derecha, sección **Input**:
- **La competencia:** tiene que aparecer, con `train_data.jsonlines` y `test_data.jsonlines`. Si no aparece: **+ Add Input** → pestaña *Competitions* → buscala → **Add**.
- **El dataset de folds:** **+ Add Input** → pestaña *Your Work* / *Datasets* → buscá `folds-tp3` → **Add**.

No importa en qué carpeta quede cada archivo: el notebook los busca solo dentro de `/kaggle/input`.

## 4. Activar la GPU e internet

En el panel de la derecha, sección **Settings** (o el menú **⋮** → *Accelerator*):
1. **Accelerator:** elegí **GPU T4 x2** (o **GPU T4**; también sirve *GPU P100*). El notebook usa una sola GPU.
2. **Internet:** activalo (**On**). Hace falta para descargar el modelo BETO.

La primera vez, Kaggle puede pedir **verificar tu cuenta con el número de teléfono** para habilitar la GPU y el internet: te manda un código por SMS. Es un paso único.

Kaggle da una cuota semanal de horas de GPU (alrededor de 30; se ve en *Settings* o en tu perfil). Esta corrida usa menos de 1 hora.

## 5. Correrlo en segundo plano

No hace falta dejar la computadora prendida:
1. Arriba a la derecha, **Save Version**.
2. Elegí **Save & Run All (Commit)** y tocá **Save**.
3. Kaggle ejecuta el notebook completo en sus servidores. Podés cerrar el navegador y apagar la computadora.

Para ver cómo va: entrá al notebook (desde **Code** → *Your Work*) y mirá la versión en ejecución; el log muestra el avance por fold y por época. Cuando termina, la versión queda en estado *Complete* (o *Error*, ver el punto 7).

> Si preferís ver la salida en vivo, también podés ejecutar celda por celda con el botón ▶ (o *Run All*), pero en ese caso tenés que dejar la pestaña abierta y la computadora prendida.

## 6. Descargar los resultados

1. Abrí la versión terminada del notebook → pestaña o sección **Output**.
2. Descargá estos dos archivos (botón de descarga al lado de cada uno):
   - `finetune_titulo_oof.csv`: columnas fila, fold, proba_used (70.000 filas);
   - `finetune_titulo_test.csv`: columnas fila, proba_used (30.000 filas).
3. Guardalos en la carpeta del proyecto, en **`predicciones/finetune/`** (creala si no existe).

También están los archivos por fold (`fold_k_oof.csv`, `fold_k_test.csv`, `fold_k_info.json`). No hacen falta, salvo que la corrida se corte (ver abajo).

## 7. Si algo falla

| Problema | Qué hacer |
|---|---|
| `RuntimeError: No hay GPU` | El *Accelerator* quedó en *None*. Activá la GPU (punto 4) y volvé a correr. |
| `FileNotFoundError: No se encontró ...` | Falta un dato de entrada. El mensaje lista lo que hay en `/kaggle/input`; agregá la competencia o el dataset `folds-tp3` (punto 3). |
| Falla al descargar el modelo (`OSError`, `Connection error`, `not a valid model identifier`) | Revisá que **Internet** esté en *On*. Si sigue fallando, cambiá `MODELO` en la primera celda de código por `"bertin-project/bertin-roberta-base-spanish"` o `"dccuchile/bert-base-spanish-wwm-cased"`. (`PlanTL-GOB-ES/roberta-base-bne` ya no sirve: el repositorio fue discontinuado y no tiene los pesos.) |
| `CUDA out of memory` | Bajá `BATCH = 64` a `BATCH = 32` en la primera celda de código. Va a tardar un poco más. |
| La sesión se cortó a mitad de camino | Si estabas ejecutando celda por celda, volvé a ejecutar: los folds ya guardados en `/kaggle/working` se leen y no se reentrenan. Una versión nueva (*Save & Run All*) empieza de cero; descargá antes los `fold_k_*` de la versión cortada si querés conservarlos. |
| `folds.csv no corresponde a train_data.jsonlines` | Subiste otro archivo de folds. Volvé a subir `kaggle/folds.csv` del proyecto. |

## Opcional: título + warranty

En la primera celda de código, cambiá `TEXTO = "titulo"` por `TEXTO = "titulo_warranty"` para entrenar con el título y el texto de garantía juntos. Los archivos de salida se llaman igual: renombralos al descargarlos (por ejemplo, `finetune_texto_oof.csv`) para no pisar los del título solo.

---

# Segunda tanda: `finetune_titulos_v2.ipynb`

Los pasos son los mismos que arriba (dataset, notebook desde la competencia, GPU e internet, *Save & Run All*). Lo que cambia:

**Qué corre.** Dos configuraciones, una detrás de otra (lista `CONFIGURACIONES` en la primera celda de código):
- a) `beto_titulo_warranty`: BETO uncased con título + `" [SEP] "` + warranty, `max_length` 96, semilla 7;
- b) `beto_cased_titulo`: BETO cased con el título, semilla 42. Si no descarga, usa `bertin-project/bertin-roberta-base-spanish` y los archivos se llaman `bertin_titulo`.

**Cuánto tarda.** ~1,5-2,5 horas con una T4: a) ~60-80 minutos, b) ~35-50. Al empezar muestra una estimación previa, y otra después del primer fold con el tiempo medido. Si alguna supera 8,5 horas, avisa.

**Datos de entrada.**
- Igual que antes: la competencia y el dataset con `folds.csv`. El notebook nuevo se importa con *File → Import Notebook* eligiendo `kaggle/finetune_titulos_v2.ipynb`.
- **Opcional:** subí también `predicciones/finetune/finetune_titulo_oof.csv` (las predicciones de BETO título de la primera corrida). Puede ir en el mismo dataset `folds-tp3`: *New Version* → agregás el archivo. También puede ir en un dataset nuevo, que agregás con **+ Add Input**. Si está, el notebook muestra la correlación de cada configuración con BETO título y el promedio de los dos. Si no está, se saltea esa parte.

**Archivos que genera** en `/kaggle/working`, uno por configuración:
- `finetune_beto_titulo_warranty_oof.csv` y `finetune_beto_titulo_warranty_test.csv`;
- `finetune_beto_cased_titulo_oof.csv` y `finetune_beto_cased_titulo_test.csv` (o `finetune_bertin_titulo_*` si se usó BERTIN).

Los `_oof.csv` tienen las columnas fila, fold y proba_used (70.000 filas). Los `_test.csv` tienen fila, desde 0, y proba_used (30.000 filas). También están los archivos por fold (`{nombre}_fold_k_*`), que solo hacen falta para retomar.

**Dónde van.** Descargá los archivos finales a **`predicciones/finetune/`** del proyecto. Si Kaggle los baja con extensión `.txt`, renombralos a `.csv` (por ejemplo `finetune_beto_cased_titulo_oof.txt` → `finetune_beto_cased_titulo_oof.csv`). El contenido es el mismo.

**Si se corta.**
- Si estás ejecutando celda por celda en la misma sesión, volvé a ejecutar. Las configuraciones con sus dos archivos finales completos se saltean, y los folds ya guardados no se reentrenan.
- Una versión nueva (*Save & Run All*) empieza con `/kaggle/working` vacío. Antes de lanzarla, descargá lo terminado de la versión cortada y sacá de `CONFIGURACIONES` lo que ya tengas.

**Si un modelo no descarga.** El notebook prueba el siguiente candidato de la lista; si fallan todos, saltea esa configuración y sigue con la próxima. Revisá que **Internet** esté en *On*.

---

# TabICLv2: `tabicl.ipynb`

TabICLv2 (Inria, paquete `tabicl`) es un modelo fundacional para datos tabulares: predice usando las filas de entrenamiento como contexto, sin entrenarse con este dataset. Corre sobre las mismas 147 variables del LightGBM 15b, con la validación anidada. Necesita GPU.

**No usa los datos de la competencia**, solo el dataset `tabicl_datos`.

## 1. Subir `tabicl_datos` como dataset privado

1. **Datasets** → **+ New Dataset**.
2. Arrastrá **todos los archivos** de la carpeta `kaggle/tabicl_datos/` del proyecto (49 MB):
   - los 10 `anidada_fold{k}_train.parquet` / `anidada_fold{k}_valid.parquet`;
   - `estandar_train.parquet` y `estandar_test.parquet`;
   - `columnas.json`.
3. Título: por ejemplo `tabicl-datos-tp3`. Visibilidad **Private** → **Create**.

Si alguna vez cambian las variables, se regeneran con `APTP3.venv/Scripts/python.exe kaggle/exportar_tabicl_datos.py` (~1-2 minutos) y se sube una versión nueva del dataset (*New Version*).

## 2. Crear el notebook e importar `tabicl.ipynb`

1. **Code** → **+ New Notebook**. Puede ser desde cualquier lado: no hace falta la competencia.
2. **File** → **Import Notebook** → elegí `kaggle/tabicl.ipynb` → **Import**.
3. Panel derecho, **Input** → **+ Add Input** → *Your Work* / *Datasets* → `tabicl-datos-tp3` → **Add**.

El notebook busca los archivos solo, en cualquier carpeta dentro de `/kaggle/input`.

## 3. Activar GPU e internet

En **Settings**:
- **Accelerator:** **GPU T4 x2** (o **GPU T4**). Se usa una sola GPU.
- **Internet:** **On**. Hace falta para instalar `tabicl` y para que el modelo se descargue solo de Hugging Face (~100 MB, sin token).

## 4. Correrlo en segundo plano

**Save Version** → **Save & Run All (Commit)** → **Save**. Podés cerrar el navegador.

**Qué corre**, una configuración detrás de otra:
- `todas`: las 147 variables, `n_estimators=8`;
- `top100`: las 100 más importantes del LightGBM 15b, `n_estimators=8`;
- `top100_n16`: las mismas 100 con `n_estimators=16`, solo si el primer fold de `todas` tarda menos de 5 minutos.

Cada una hace los 5 folds anidados y después el test.

**Cuánto tarda.** No hay una medición previa en un dataset de este tamaño: estimado **~1-7 horas**.
- Después del primer fold, el log muestra una estimación con el tiempo medido.
- Si supera 8,5 horas, avisa.

## 5. Descargar los resultados

En la versión terminada → **Output**, descargá por cada configuración:
- `tabicl_{config}_oof.csv`: fila, fold, proba_used (70.000 filas);
- `tabicl_{config}_test.csv`: fila (desde 0), proba_used (30.000 filas).

Guardalos en **`predicciones/tabicl/`** del proyecto (creala si no existe). Si bajan como `.txt`, renombralos a `.csv`.

Los `tabicl_{config}_fold_{k}.csv` / `.json` solo hacen falta para retomar. El `.json` de cada fold guarda las métricas, el tiempo y el intento de memoria usado.

## 6. Si algo falla

| Problema | Qué hacer |
|---|---|
| `RuntimeError: No hay GPU` | Activá la GPU (punto 3) y volvé a correr. |
| `FileNotFoundError: No se encontró ...` | Falta el dataset. El mensaje lista lo que hay en `/kaggle/input`; agregá `tabicl-datos-tp3` (punto 2). |
| Falla `pip install` o la descarga del modelo (`ConnectionError`, `HfHubHTTPError`, `LocalEntryNotFoundError`) | **Internet** tiene que estar en *On*. Si Hugging Face está caído, esperá y volvé a correr. El modelo es `jingang/TabICL`, archivo `tabicl-classifier-v2-20260212.ckpt`. |
| Mensajes `Error de memoria ... Reintento con ...` | No hace falta hacer nada: el notebook reintenta solo con `batch_size` 4 y después 2, con `offload_mode="auto"`. Tarda más. |
| `CUDA out of memory` aun con `batch_size` 2 | Sacá `todas` de `CONFIGURACIONES` y corré solo `top100` (menos columnas, menos memoria). |
| El kernel se reinicia solo (*Kernel died*) | Se quedó sin memoria RAM al pasar datos de la GPU a la CPU. Mismo arreglo: solo `top100`. |
| La sesión se cortó | Una versión nueva empieza con `/kaggle/working` vacío. Descargá lo terminado de la versión cortada (los `_oof`, `_test` y los `fold_k`) y sacá de `CONFIGURACIONES` lo que ya tengas completo. Si ejecutás celda por celda en la misma sesión, los folds guardados se saltean solos. |

# TabICLv2, segunda corrida: `tabicl_v2.ipynb`

Igual que la corrida anterior (mismos pasos 3, 4 y 6), pero con las 182 variables definitivas: las de `variables_21.json` (LightGBM 21b = 19b + forma de las fotos + `te_tamano_foto`). Configuraciones `todas` (182) y `top100`.

1. **Datos:** subí **todos los archivos** de `kaggle/tabicl_datos_v2/` (88,5 MB, 13 archivos) como **dataset nuevo** (*Create* → *New Dataset*, por ejemplo `tabicl-datos-v2-tp3`, *Private*). También sirve como versión nueva del anterior (*New Version* de `tabicl-datos-tp3`), pero entonces el notebook viejo ya no puede volver a correr con las 147 variables. Si hay que regenerarlos: `APTP3.venv/Scripts/python.exe kaggle/exportar_tabicl_datos_v2.py` (~6 minutos, con verificación).
2. **Notebook:** **File** → **Import Notebook** → `kaggle/tabicl_v2.ipynb`. En **Input**, agregá `tabicl-datos-v2-tp3` (podés quitar `tabicl-datos-tp3`). El notebook usa la carpeta cuyo `columnas.json` dice `"version": "tabicl_datos_v2"`, así que no se confunde si los dos datasets están agregados.
3. **Correrlo** igual que antes: GPU T4, Internet On, **Save Version** → *Save & Run All (Commit)*.
4. **Descargar** de *Output* `tabicl_v2_todas_oof.csv`, `tabicl_v2_todas_test.csv`, `tabicl_v2_top100_oof.csv` y `tabicl_v2_top100_test.csv` a **`predicciones/tabicl/`** (si bajan como `.txt`, renombralos a `.csv`). Tienen otro nombre que los de la primera corrida (`tabicl_{config}_*`), así que no los pisan.

## Si se corta: retomar desde la versión anterior

La versión 1 se quedó sin RAM en el test de `todas` ("Your notebook tried to allocate more memory than is available", *Kernel died*), con los 5 folds de `todas` ya terminados. El notebook ahora hace el test con los embeddings por columna en disco y `batch_size` 2 (ver la sección "Modelo"). Además, en los folds sigue con el intento que funcionó, así que ya no pierde uno por fold.

Para no repetir lo que ya está hecho:
1. En la versión que falló, pestaña **Output**: descargá los archivos `tabicl_v2_*` (`tabicl_v2_todas_fold_1.csv` … `_fold_5.csv`, sus `.json` y `tabicl_v2_todas_oof.csv`).
2. Subilos como dataset privado nuevo (por ejemplo `tabicl-v2-previo`) y agregalo como **Input** del notebook, junto con `tabicl-datos-v2-tp3`. Si Kaggle ofrece agregar directamente la salida de esa versión (*Add Input* → *Your Work* → el notebook), también sirve.
3. Volvé a importar `kaggle/tabicl_v2.ipynb` (**File** → **Import Notebook**, reemplaza el código) y **Save Version** → *Save & Run All (Commit)*. Al empezar copia esos archivos a `/kaggle/working` y lo muestra en la celda de datos ("Archivos de una versión anterior copiados…"). Los folds de `todas` se saltean y arranca por su test.
