# Objetivo del MVP

## Propósito

Construir una API local para que investigadores con conocimientos técnicos
analicen grabaciones de campo y obtengan una estimación de si el audio proviene
de una fuente acústica física o si fue generado o alterado electrónicamente.

## Alcance inicial

- Recibir un archivo de audio por solicitud en formato WAV, MP3 o FLAC.
- Admitir grabaciones de varios minutos.
- Clasificar el audio en una de estas opciones:
  - **Físico:** sonido capturado de una fuente acústica real, como una entidad
    biológica, un fenómeno natural o un evento físico.
  - **Digital:** sonido creado electrónicamente, incluidos los sintetizados y
    los que incorporan efectos digitales.
  - **Incierto:** el modelo no tiene confianza suficiente o el audio real está
    tan procesado que no es posible asignarle una clase de forma fiable.
- Priorizar la detección de audios digitales, midiendo primero un baseline
  antes de establecer objetivos de rendimiento.
- Usar TensorFlow para el modelo y FastAPI para exponer la API.
- Mantener la primera versión en un entorno local; no incluir interfaz web,
  despliegue público ni procesamiento por lotes.

## Datos y evaluación

Como todavía no hay audios disponibles, el trabajo inicial combinará la
búsqueda de conjuntos de datos públicos con la recopilación de grabaciones
propias. Se deberán documentar la procedencia y las etiquetas de los ejemplos.

La evaluación deberá separar las grabaciones —y, cuando sea posible, sus
fuentes— entre entrenamiento y prueba para evitar que fragmentos del mismo
audio aparezcan en ambos conjuntos. Además de las métricas generales, se
prestará especial atención a la detección de audios digitales y a los falsos
positivos. Los casos dudosos podrán devolverse como **inciertos** en vez de
forzar una clasificación.

## Fuera del alcance

- Garantizar una determinación concluyente del origen de cualquier audio.
- Procesar varios archivos en una sola solicitud.
- Ofrecer una aplicación web o un servicio público.
- Clasificar en subcategorías más específicas que físico, digital e incierto.

El resultado del MVP es una herramienta experimental de apoyo a la
investigación. Su respuesta debe interpretarse junto con las limitaciones del
conjunto de datos y del modelo.
