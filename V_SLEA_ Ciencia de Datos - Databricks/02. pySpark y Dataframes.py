# Databricks notebook source
# MAGIC %md
# MAGIC ## pySpark
# MAGIC Como vimos en la lección anterior, [pySpark](http://spark.apache.org/docs/latest/api/python/) es la interfaz de Python para Apache Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC ##DataFrames
# MAGIC Un DataFrame es la estructura de datos con la que más interactuaremos al utilizar pySpark.  

# COMMAND ----------

# MAGIC %md
# MAGIC ###Creación
# MAGIC Un DataFrame puede crearse de diferentes maneras:
# MAGIC 1. Especificando los datos que contendrá y el esquema (opcional) de forma explícita, esto es útil para datos estáticos o pruebas.
# MAGIC 2. Leyendo otra estructura de datos, que generalmente será un archivo o datos provenientes de una base de datos relacional.
# MAGIC 3. A partir de otro DataFrame.
# MAGIC 4. Con los datos de una tabla Spark, cuyos metadatos están almacenados dentro de Databricks.
# MAGIC
# MAGIC Veamos cada uno de estos ejemplos.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creación manual

# COMMAND ----------

datos = [{"id": 1, "monto": 121.44, "es_valido": True},
        {"id": 2, "monto": 300.01, "es_valido": False},
        {"id": 3, "monto": 10.99,  "es_valido": None}]
df_manual = spark.createDataFrame(datos)

df_manual.show()

# COMMAND ----------

df_manual.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Como mencionábamos, un DataFrame tiene un esquema definido, en este caso inferido por Spark porque no se lo especificamos.

# COMMAND ----------

df_manual.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Lectura de un archivo
# MAGIC Leemos un csv de un repositorio público. Veremos más detalles del mismo, y de las opciones que utilizamos a continuación, en el próximo módulo.

# COMMAND ----------

ruta_archivo = "/databricks-datasets/definitive-guide/data/retail-data/all/online-retail-dataset.csv"

df_archivo = (spark.read
  .option("sep", ",")
  .option("header", True)
  .option("inferSchema", True)
  .csv(ruta_archivo)
)

# COMMAND ----------

display(df_archivo)

# COMMAND ----------

# MAGIC %md
# MAGIC ####A partir de otro DataFrame
# MAGIC Usamos *df_archivo* para generar un nuevo DF.

# COMMAND ----------

df_archivo_filtrado = (df_archivo
  .select("InvoiceNo", "Quantity")
  .where("Quantity == 6")
                      )
                       
display(df_archivo_filtrado)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tabla Spark
# MAGIC Si tenemos creadas [tablas](https://docs.databricks.com/data/tables.html) en Databricks, podemos crear un DataFrame a partir de estas.    

# COMMAND ----------

# Primero creamos una tabla y luego la leemos
dbutils.fs.rm("/user/hive/warehouse/prueba", True)
df_archivo_filtrado.write.saveAsTable("prueba")
# df_tabla = spark.table("prueba")
# display(df_tabla)

# COMMAND ----------

dbutils.fs.ls("/user/hive/warehouse/")


# COMMAND ----------

try:
    print(dbutils.fs.ls("/user/hive/warehouse/prueba"))
except Exception as e:
    print("La carpeta no existe o no tienes permisos")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Otras operaciones
# MAGIC A continuación, mostramos sólo algunas de las operaciones que se pueden realizar con pySpark.

# COMMAND ----------

# Agregado de columnas
from pyspark.sql.functions import col
df_monto = df_archivo.withColumn("monto", col("Quantity")*col("UnitPrice"))

# Seleccion y descarte de columnas
df=df_monto.select("Quantity","UnitPrice","monto")
display(df)
display(df.drop("Quantity"))

# Ordenado
display(df.sort(col("monto").desc()))

# COMMAND ----------

display(df_tabla.select("*").limit(1))
display(spark.sql("SELECT * FROM prueba LIMIT 1"))
