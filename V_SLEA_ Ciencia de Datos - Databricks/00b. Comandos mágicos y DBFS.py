# Databricks notebook source
# MAGIC %md
# MAGIC ## Comandos mágicos
# MAGIC Son comandos especiales que nos permiten realizar acciones especiales en las notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Markdown
# MAGIC Con el comando mágico *%md*, podemos usar la celda de la notebook para documentar utilizando el lenguaje [Markdown](https://www.markdownguide.org/basic-syntax/)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DBFS  (Databricks File System)
# MAGIC Es un sistema de archivos distribuido que se monta en un Workspace de Databricks.  
# MAGIC Sirve como abstracción de los objetos almacenados en los servicios de almacenamiento de las distintas nubes.  
# MAGIC De esta manera, podremos ir *montando* nuestros Datalake y accederlos desde DBFS.
# MAGIC Además, por defecto vamos a encontrar estos directorios:
# MAGIC - /databricks-datasets: datasets públicos de ejemplo.
# MAGIC - /FileStore: archivos que importemos de forma manual.
# MAGIC - /databricks-results: archivos generados al descargar resutlados de una consulta.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. **`DBFS en Databricks`**:
# MAGIC    - DBFS (Databricks File System) es un sistema de archivos distribuido y escalable diseñado específicamente para la plataforma de análisis de datos en la nube, Databricks. Proporciona un espacio de almacenamiento persistente y de alto rendimiento para datos y artefactos utilizados en análisis y procesamiento de datos.
# MAGIC
# MAGIC 2. **`Características Clave`**:
# MAGIC    - **Integración con Databricks**:
# MAGIC      - DBFS está integrado de forma nativa en la plataforma de Databricks, lo que permite acceder y gestionar datos directamente desde los Notebooks de Databricks, la línea de comandos y las APIs de Databricks.
# MAGIC    - **Persistencia y escalabilidad**:
# MAGIC      - Ofrece persistencia y escalabilidad para los datos utilizados en análisis de datos en la nube. Los datos almacenados en DBFS están disponibles de manera persistente incluso después de que se detengan las instancias de clúster.
# MAGIC    - **Soporte para diferentes tipos de datos**:
# MAGIC      - Puede manejar varios tipos de datos, incluyendo archivos de datos estructurados (por ejemplo, CSV, JSON, Parquet), archivos binarios, librerías y artefactos de código, entre otros.
# MAGIC    - **Interoperabilidad**:
# MAGIC      - Compatible con otros sistemas de almacenamiento en la nube como Amazon S3, Azure Blob Storage y Google Cloud Storage, lo que facilita el acceso e integración con datos almacenados en estos sistemas desde la plataforma de Databricks.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### fs
# MAGIC Siguiendo con los comandos mágicos, con fs accedemos a DBFS, luego del mismo especificamos el comando a utilizar, por ejemplo *ls* para listar archivos.

# COMMAND ----------

# MAGIC %fs ls "/databricks-datasets/definitive-guide/data/activity-data/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### dbutils
# MAGIC Son [utilidades](https://docs.databricks.com/dev-tools/databricks-utils.html#file-system-utility-dbutilsfs) provistas por Databricks para ejecutar diversas tareas.  
# MAGIC Una de ellas es el acceso a [DBFS](https://docs.databricks.com/dev-tools/databricks-utils.html#file-system-utility-dbutilsfs):

# COMMAND ----------

# Con el comando help vemos las distintas opciones que tenemos disponibles.
dbutils.help()

# COMMAND ----------

# MAGIC %md
# MAGIC Por ejemplo, con *fs head* vemos las primeras lineas de un archivo.  

# COMMAND ----------

dbutils.fs.head("/databricks-datasets/definitive-guide/data/retail-data/all/online-retail-dataset.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC El comando anterior podría ejecutarse con el magic *fs*:

# COMMAND ----------

# MAGIC %fs head /databricks-datasets/definitive-guide/data/retail-data/all/online-retail-dataset.csv

# COMMAND ----------

dbutils.fs.help("mkdirs")

# COMMAND ----------

# MAGIC %md
# MAGIC # Ejemplos de `dbutils` en Databricks
# MAGIC
# MAGIC ## 1. Manejo de Archivos (`dbutils.fs`)
# MAGIC
# MAGIC ### Listar archivos en un directorio:
# MAGIC ```python
# MAGIC dbutils.fs.ls("/mnt/data")
# MAGIC ```
# MAGIC
# MAGIC ### Crear un directorio:
# MAGIC ```python
# MAGIC dbutils.fs.mkdirs("/mnt/data/nuevo_directorio")
# MAGIC ```
# MAGIC
# MAGIC ### Copiar un archivo:
# MAGIC ```python
# MAGIC dbutils.fs.cp("/mnt/data/origen.csv", "/mnt/data/destino.csv")
# MAGIC ```
# MAGIC
# MAGIC ### Mover un archivo:
# MAGIC ```python
# MAGIC dbutils.fs.mv("/mnt/data/origen.csv", "/mnt/data/destino.csv")
# MAGIC ```
# MAGIC
# MAGIC ### Eliminar un archivo o directorio:
# MAGIC ```python
# MAGIC dbutils.fs.rm("/mnt/data/archivo.csv", recurse=True)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Manejo de Widgets (`dbutils.widgets`)
# MAGIC
# MAGIC ### Crear un widget de texto:
# MAGIC ```python
# MAGIC dbutils.widgets.text("parametro1", "valor_predeterminado", "Ingrese un valor")
# MAGIC ```
# MAGIC
# MAGIC ### Obtener el valor del widget:
# MAGIC ```python
# MAGIC valor = dbutils.widgets.get("parametro1")
# MAGIC print(valor)
# MAGIC ```
# MAGIC
# MAGIC ### Eliminar un widget:
# MAGIC ```python
# MAGIC dbutils.widgets.remove("parametro1")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Manejo de Secretos (`dbutils.secrets`)
# MAGIC
# MAGIC ### Listar alcances de secretos disponibles:
# MAGIC ```python
# MAGIC dbutils.secrets.listScopes()
# MAGIC ```
# MAGIC
# MAGIC ### Listar secretos dentro de un alcance:
# MAGIC ```python
# MAGIC dbutils.secrets.list("mi_alcance")
# MAGIC ```
# MAGIC
# MAGIC ### Obtener un secreto:
# MAGIC ```python
# MAGIC dbutils.secrets.get(scope="mi_alcance", key="mi_secreto")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Ejecución de Scripts y Comandos (`dbutils.notebook`)
# MAGIC
# MAGIC ### Ejecutar otro notebook:
# MAGIC ```python
# MAGIC dbutils.notebook.run("ruta_del_notebook", 60, {"param1": "valor1"})
# MAGIC ```
# MAGIC > *Donde `60` es el tiempo máximo de ejecución en segundos.*
# MAGIC
# MAGIC ### Salir de un notebook con un mensaje de salida:
# MAGIC ```python
# MAGIC dbutils.notebook.exit("Ejecución finalizada con éxito")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5. Interacción con Clústeres (`dbutils.library`)
# MAGIC
# MAGIC ### Instalar una librería desde PyPI:
# MAGIC ```python
# MAGIC dbutils.library.installPyPI("pandas")
# MAGIC ```
# MAGIC
# MAGIC ### Reiniciar el clúster después de instalar librerías:
# MAGIC ```python
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC
