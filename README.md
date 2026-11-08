# deepfake_audio_detector
Sistema de clasificación de audio impulsado por IA para la materia técnicas de procesamiento digital de imágenes de la tecnicatura superior en ciencia de datos e IA

## Entorno de desarrollo

El proyecto usa UV y Python 3.11. Desde la raíz del repositorio, ejecutá:

```powershell
uv sync
```

UV crea el entorno virtual `.venv` e instala las dependencias fijadas en
`pyproject.toml` y `uv.lock`. Para ejecutar comandos dentro del entorno sin
activarlo manualmente:

```powershell
uv run python --version
```

En Windows, TensorFlow requiere `tensorflow-intel` como backend. Se excluye
`tensorflow-io-gcs-filesystem`, que no dispone de una rueda para Windows y no es
necesario para analizar archivos de audio locales.

## API local

Iniciá el servidor con:

```powershell
uv run uvicorn main:app --reload
```

La documentación interactiva de FastAPI queda disponible en
<http://127.0.0.1:8000/docs>. El endpoint `POST /predict` acepta un archivo
WAV, MP3, OGG o FLAC. Para obtener predicciones se necesita un modelo entrenado
en `models/audio_model.keras`; si aún no está disponible, la API responde con
HTTP 503.

## Datos de entrenamiento

Guardá los audios originales en `data/raw/fisico/` o `data/raw/digital/` según
su etiqueta, y registrá cada archivo en `data/metadata.csv`. La columna
`relative_path` debe apuntar al audio desde la raíz del repositorio. `label`
debe ser `fisico` o `digital`, y `source_id` es obligatorio: usa el mismo ID
para grabaciones que comparten una fuente, para evitar que queden a la vez en
entrenamiento y validación. Los audios originales y los datos derivados de
`data/processed/` se excluyen de Git.

Con ejemplos etiquetados de ambas clases, entrená el modelo con:

```powershell
uv run python -m training.train
```

El módulo `training/` contiene el script de entrenamiento y las funciones de
preprocesamiento de audio compartidas con el clasificador. El script divide las
grabaciones por `source_id`, informa exactitud y recall digital a nivel de
grabación para el conjunto de validación y guarda el modelo en
`models/audio_model.keras`. Se requieren al menos dos fuentes distintas por
clase para poder separar entrenamiento y validación; con tan pocos datos, las
métricas no serán concluyentes. Se pueden ajustar épocas, tamaño de lote,
semilla, CSV, raíz de datos y ruta de salida con `--epochs`, `--batch-size`,
`--seed`, `--metadata`, `--data-root` y `--output`.
