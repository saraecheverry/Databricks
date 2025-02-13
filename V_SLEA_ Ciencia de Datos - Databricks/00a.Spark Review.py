# Databricks notebook source
# MAGIC %md <i18n value="1108b110-983d-4034-9156-6b95c04dc62c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Review de Spark
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) En esta lección:<br>
# MAGIC  - Creación de dataframes de Spark
# MAGIC  - Entendimiento del UI de Spark
# MAGIC  - Usando en Cache 
# MAGIC  - Relaciones entre Spark DF y Pandas

# COMMAND ----------

# MAGIC %md
# MAGIC ## Arquitectura de Spark

# COMMAND ----------

# MAGIC %md <i18n value="890d085b-9058-49a7-aa15-bff3649b9e05"/>
# MAGIC <img src="https://lh3.googleusercontent.com/d/1mUpZ6pb-L_axuKnTdaJhgq3E4lSpWopw">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Arquitectura de Apache Spark
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Apache Spark tiene una arquitectura distribuida diseñada para el procesamiento de grandes volúmenes de datos de manera eficiente. Su estructura básica se compone de varios componentes clave.
# MAGIC
# MAGIC ## 1. Componentes principales de la arquitectura de Spark
# MAGIC
# MAGIC ### 1.1 Driver Program (Programa Controlador)
# MAGIC Es el punto de entrada de la aplicación Spark. Aquí se define la lógica del programa y se coordina la ejecución de tareas en el clúster.
# MAGIC - Crea un **SparkContext** (o **SparkSession** en Spark 2+).
# MAGIC - Divide el código en **jobs** (trabajos) y los envía a los ejecutores.
# MAGIC - Monitorea la ejecución y maneja fallos.
# MAGIC
# MAGIC ### 1.2 Cluster Manager (Administrador del Clúster)
# MAGIC Se encarga de gestionar los recursos (CPU, memoria) en los nodos del clúster. Spark puede trabajar con diferentes gestores de clústeres:
# MAGIC - **Standalone** (modo nativo de Spark).
# MAGIC - **YARN** (usado en Hadoop).
# MAGIC - **Mesos** (usado en entornos más avanzados).
# MAGIC - **Kubernetes** (para entornos en contenedores).
# MAGIC
# MAGIC ### 1.3 Executors (Ejecutores)
# MAGIC Son procesos que se ejecutan en los **worker nodes** (nodos de trabajo) y realizan el procesamiento real de los datos.
# MAGIC - Cada ejecutor recibe tareas asignadas por el **driver**.
# MAGIC - Mantienen datos en caché para mejorar el rendimiento.
# MAGIC - Una vez terminadas las tareas, reportan los resultados al **driver**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Flujo de ejecución de un programa Spark
# MAGIC
# MAGIC 1. **El driver lanza un SparkContext/SparkSession.**
# MAGIC 2. **El Cluster Manager asigna recursos y lanza ejecutores.**
# MAGIC 3. **El driver divide el trabajo en "Jobs", que se dividen en "Stages" y "Tasks".**
# MAGIC 4. **Los ejecutores procesan los datos en paralelo y envían los resultados al driver.**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Módulos principales de Spark
# MAGIC
# MAGIC Spark tiene varios módulos integrados para diferentes tareas:
# MAGIC - **Spark Core**: Procesamiento básico de datos.
# MAGIC - **Spark SQL**: Consultas estructuradas con SQL y DataFrames.
# MAGIC - **Spark Streaming**: Procesamiento en tiempo real.
# MAGIC - **MLlib**: Aprendizaje automático.
# MAGIC - **GraphX**: Procesamiento de grafos.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Resilient Distributed Dataset (RDD) - Concepto Clave
# MAGIC
# MAGIC Spark usa una estructura de datos llamada **RDD** (Resilient Distributed Dataset), que permite:
# MAGIC - Almacenar datos distribuidos en memoria o disco.
# MAGIC - Tolerancia a fallos mediante reconstrucción automática.
# MAGIC - Transformaciones y acciones para manipular datos de forma eficiente.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md <i18n value="df081f79-6894-4174-a554-fa0943599408"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Spark DataFrame

# COMMAND ----------

from pyspark.sql.functions import col, rand

df = (spark.range(1, 10000)
      .withColumn("id", (col("id") / 1000).cast("integer"))
      .withColumn("v", rand(seed=1)))

# COMMAND ----------

df.show(5)

# COMMAND ----------

display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC El método createTempView en PySpark se utiliza para crear una vista temporal a partir de un DataFrame. Esta vista temporal es similar a una tabla en una base de datos, pero solo existe durante la sesión actual de Spark.

# COMMAND ----------

# df.createTempView("tabla")

# Crear una vista temporal llamada 'tabla'
df.createOrReplaceTempView("tabla")

# Ejecutar una consulta SQL sobre la vista temporal
result = spark.sql("SELECT * FROM tabla LIMIT 5")

# Mostrar los resultados de la consulta
result.show()

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT COUNT(*) FROM tabla

# COMMAND ----------

# MAGIC %md
# MAGIC ¿Por que no vemos ningun job con las lineas anteriores? -> En realidad no hemos hecho ninguna operación sobre los datos. Spark no tiene necesidad de ejecutar nada sobre el cluster

# COMMAND ----------

# MAGIC %md
# MAGIC ## Procesos en Spark

# COMMAND ----------

# MAGIC %md <i18n value="890d085b-9058-49a7-aa15-bff3649b9e05"/>
# MAGIC <img src="https://lh3.googleusercontent.com/d/15yWwqeZqZp9Vd7w_8kSteUAs3M3NdreG">

# COMMAND ----------

display(df.sample(.001))

# COMMAND ----------

# MAGIC %md <i18n value="6eadef21-d75c-45ba-8d77-419d1ce0c06c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Views
# MAGIC
# MAGIC Interoperabilidad entre dataframes y tablas. 
# MAGIC La interoperabilidad entre DataFrames y tablas se refiere a la capacidad de trabajar y manipular datos de manera fluida entre diferentes formatos y estructuras dentro de un entorno de procesamiento de datos, como PySpark. En PySpark, los datos se pueden representar tanto como DataFrames como tablas SQL, y la interoperabilidad permite a los usuarios aprovechar las ventajas de ambos enfoques sin tener que preocuparse por la conversión manual entre estos formatos.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ejemplo 1 

# COMMAND ----------

df.createOrReplaceTempView("df_temp")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM df_temp LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ejemplo 2

# COMMAND ----------

df.write.saveAsTable("permanent_table")

# COMMAND ----------

result = spark.sql("SELECT * FROM permanent_table WHERE id == 1")
result.show()

# COMMAND ----------

# Elimina la tabla utilizando Spark SQL
spark.sql("DROP TABLE IF EXISTS permanent_table")

# COMMAND ----------

# MAGIC %md <i18n value="2593e6b0-d34b-4086-9fed-c4956575a623"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Count
# MAGIC
# MAGIC Cuantas observaciones tenemos? 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM df_temp LIMIT 10

# COMMAND ----------

_sqldf.count()

# COMMAND ----------

# MAGIC %md
# MAGIC En el contexto de _sqldf, que generalmente se refiere al resultado de una consulta SQL ejecutada a través de spark.sql

# COMMAND ----------


# # Ejecutar una consulta SQL sobre la vista temporal y asignarla a _sqldf
# _sqldf = spark.sql("SELECT * FROM df_temp LIMIT 10")

# # Contar el número de filas en el DataFrame _sqldf
# row_count = _sqldf.count()

# # Mostrar el número de filas
# print(f"Number of rows: {row_count}")

# COMMAND ----------

display(_sqldf)

# COMMAND ----------

display(_sqldf.sort("v"))

# COMMAND ----------

# MAGIC %md
# MAGIC El método repartition en PySpark se utiliza para redistribuir los datos en un DataFrame en un número específico de particiones. Cuando ejecutas df.repartition(10), estás repartiendo el DataFrame df en 10 particiones.
# MAGIC
# MAGIC Esto puede ser útil por varias razones:
# MAGIC
# MAGIC - Optimización de Performance: Cambiar el número de particiones puede mejorar el rendimiento de ciertas operaciones, especialmente aquellas que requieren redistribución de datos, como join, groupBy, y aggregate. Un número adecuado de particiones puede balancear la carga de trabajo entre los nodos del clúster.
# MAGIC
# MAGIC - Manejo de Datos: Aumentar o disminuir el número de particiones puede ser necesario para manejar adecuadamente la cantidad de datos en diferentes etapas del procesamiento.
# MAGIC
# MAGIC Ejecución Paralela: Con más particiones, puedes aprovechar mejor los recursos de hardware disponibles, ejecutando más tareas en paralelo.

# COMMAND ----------

df2 = df.repartition(12)

# COMMAND ----------

# MAGIC %md
# MAGIC El método df2.rdd.getNumPartitions() en PySpark se utiliza para obtener el número de particiones en el RDD subyacente de un DataFrame. Aquí df2 es un DataFrame y .rdd convierte este DataFrame en su forma de RDD (Resilient Distributed Dataset), una abstracción de bajo nivel en Spark. Luego, getNumPartitions() devuelve el número de particiones en ese RDD.
# MAGIC
# MAGIC Las particiones determinan cómo se distribuyen los datos en los nodos del clúster y afectan el paralelismo de las operaciones.

# COMMAND ----------

df2.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md <i18n value="50330454-0168-4f50-8355-0204632b20ec"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Cache
# MAGIC
# MAGIC En el caso de necesitar acceso constante repetido podemos "cachear" datos

# COMMAND ----------

# MAGIC %md
# MAGIC Explicación de `df.cache().count()` en PySpark
# MAGIC
# MAGIC El método `df.cache().count()` en PySpark combina dos operaciones: **cachear** el DataFrame y **contar** el número de filas en él. Vamos a desglosar cada parte:
# MAGIC
# MAGIC 1. **`df.cache()`**:
# MAGIC     - **Cachear** significa almacenar el DataFrame en la memoria (y posiblemente en el disco) para que las operaciones futuras en este DataFrame sean más rápidas. La primera vez que se evalúa el DataFrame después de cachearlo, Spark guarda los datos en memoria. Esto es útil cuando planeas ejecutar múltiples acciones en el mismo DataFrame, ya que evita volver a calcular los datos cada vez.
# MAGIC     
# MAGIC 2. **`df.count()`**:
# MAGIC     - **Contar** significa contar el número total de filas en el DataFrame. Esta operación desencadena la evaluación del DataFrame si no se ha hecho ya. En este caso, dado que `df.cache()` se ha llamado justo antes, los datos serán cacheados cuando `count()` los evalúe.
# MAGIC

# COMMAND ----------

df.cache().count()

# COMMAND ----------

# MAGIC %md <i18n value="7dd81880-1575-410c-a168-8ac081a97e9d"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Re-run Count

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md <i18n value="ce238b9e-fee4-4644-9469-b7d9910f6243"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Recolectar datos
# MAGIC
# MAGIC Podemos devolver datos al driver  (e.g. llamando **`.collect()`**, **`.toPandas()`**,  etc), pero debemos tener cuidado de cuantos datos podemos almacenar en la memoria de driver. Sino podemos caer en errores de tipo OOM.
# MAGIC
# MAGIC La mejor forma es explicitamente limitar el numero de observaciones.
# MAGIC
# MAGIC Claro, cuando hablamos de "recolectar datos" en el contexto de PySpark, nos referimos a la acción de traer los datos desde el clúster donde Spark está procesando los datos de vuelta al nodo maestro, también conocido como "driver", donde se ejecuta el código Python. Esto es necesario cuando queremos realizar acciones o cálculos adicionales con esos datos en el entorno local de Python.

# COMMAND ----------

pdf = df.limit(10).toPandas()

# COMMAND ----------

pdf.plot(kind = "scatter", x = "id", y = "v")

# COMMAND ----------

# MAGIC %md <i18n value="279e3325-b121-402b-a2d0-486e1cc26fc0"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Nuevo en  <a href="https://www.youtube.com/watch?v=l6SuXvhorDY&feature=emb_logo" target="_blank">Spark 3.0</a>
# MAGIC
# MAGIC * <a href="https://www.youtube.com/watch?v=jzrEc4r90N8&feature=emb_logo" target="_blank">Adaptive Query Execution</a>  https://www.databricks.com/blog/2020/05/29/adaptive-query-execution-speeding-up-spark-sql-at-runtime.html
# MAGIC   * Dynamic query optimization that happens in the middle of your query based on runtime statistics
# MAGIC     * Dynamically coalesce shuffle partitions
# MAGIC     * Dynamically switch join strategies
# MAGIC     * Dynamically optimize skew joins
# MAGIC   * Enable it with: **`spark.sql.adaptive.enabled=true`**
# MAGIC * Dynamic Partition Pruning (DPP)
# MAGIC   * Avoid partition scanning based on the query results of the other query fragments
# MAGIC * Join Hints
# MAGIC * <a href="https://www.youtube.com/watch?v=UZl0pHG-2HA&feature=emb_logo" target="_blank">Improved Pandas UDFs</a>
# MAGIC   * Type Hints
# MAGIC   * Iterators
# MAGIC   * Pandas Function API (mapInPandas, applyInPandas, etc)
# MAGIC * And many more! See the <a href="https://spark.apache.org/docs/latest/api/python/migration_guide/pyspark_2.4_to_3.0.html" target="_blank">migration guide</a> and resources linked above.

# COMMAND ----------

# MAGIC %md
# MAGIC # Adaptive Query Execution

# COMMAND ----------

# MAGIC %md
# MAGIC En notebook 'aqe-demo'

# COMMAND ----------

# MAGIC %md
# MAGIC # dynamic partition pruning

# COMMAND ----------

# MAGIC %md
# MAGIC Create Partitioned Tables

# COMMAND ----------

logger.DataSourceStrategy.name = org.apache.spark.sql.execution.datasources.DataSourceStrategy
logger.DataSourceStrategy.level = all

# COMMAND ----------

spark = SparkSession.builder \
    .appName("Dynamic Partition Pruning Example") \
    .config("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true") \
    .config("spark.sql.sources.useV1SourceList", "") \
    .getOrCreate()

# COMMAND ----------


# import org.apache.spark.sql.functions._

# spark.range(4000) \
#   .withColumn("part_id", id % 4) \
#   .withColumn("value", rand() * 100) \
#   .write \
#   .partitionBy("part_id") \
#   .saveAsTable("dpp_facts_large") 

from pyspark.sql.functions import rand, expr

spark.range(4000) \
  .withColumn("part_id", expr("id % 4")) \
  .withColumn("value", rand() * 100) \
  .write \
  .partitionBy("part_id") \
  .saveAsTable("dpp_facts_large")

# COMMAND ----------

# import org.apache.spark.sql.functions._
# spark.range(4)
#   .withColumn("name", concat(lit("name_"), 'id))
#   .write
#   .saveAsTable("dpp_dims_small")

from pyspark.sql.functions import concat, lit, col

spark.range(4) \
  .withColumn("name", concat(lit("name_"), col("id"))) \
  .write \
  .saveAsTable("dpp_dims_small")

# COMMAND ----------

facts = spark.table("dpp_facts_large")
dims = spark.table("dpp_dims_small")

# COMMAND ----------

facts.printSchema()

# COMMAND ----------

dims.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Selective Join Query

# COMMAND ----------

q = facts.join(dims) \
  .where(facts["part_id"] == dims["id"]) \
  .where(dims["id"].isin([0, 1]))

# COMMAND ----------

q.write.format("noop").mode("overwrite").save

# COMMAND ----------

q.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC # Improved Pandas UDFs

# COMMAND ----------

# MAGIC %md
# MAGIC https://www.databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType

@pandas_udf('double', PandasUDFType.SCALAR)
def pandas_plus_one(v):
    # `v` is a pandas Series
    return v.add(1)  # outputs a pandas Series

spark.range(10).select(pandas_plus_one("id")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Python Type Hints

# COMMAND ----------

def greeting(name: str) -> str:
    return 'Hello ' + name

# COMMAND ----------

# MAGIC %md
# MAGIC Proliferation of Pandas UDF Types

# COMMAND ----------



from pyspark.sql.functions import pandas_udf, PandasUDFType                                                                                                                                                                                                                                                                                                                                                                                                                              

@pandas_udf('long', PandasUDFType.SCALAR)
def pandas_plus_one(v):
    # `v` is a pandas Series
    return v + 1  # outputs a pandas Series

spark.range(10).select(pandas_plus_one("id")).show()
from pyspark.sql.functions import pandas_udf, PandasUDFType


# New type of Pandas UDF in Spark 3.0.
@pandas_udf('long', PandasUDFType.SCALAR_ITER)
def pandas_plus_one(itr):
    # `iterator` is an iterator of pandas Series.
    return map(lambda v: v + 1, itr)  # outputs an iterator of pandas Series.

spark.range(10).select(pandas_plus_one("id")).show()
from pyspark.sql.functions import pandas_udf, PandasUDFType

@pandas_udf("id long", PandasUDFType.GROUPED_MAP)
def pandas_plus_one(pdf):
# `pdf` is a pandas DataFrame
return pdf + 1  # outputs a pandas DataFrame

# `pandas_plus_one` can _only_ be used with `groupby(...).apply(...)`
spark.range(10).groupby('id').apply(pandas_plus_one).show()

# COMMAND ----------



def pandas_plus_one(v: pd.Series) -> pd.Series:
    return v + 1
def pandas_plus_one(itr: Iterator[pd.Series]) -> Iterator[pd.Series]:
    return map(lambda v: v + 1, itr)
def pandas_plus_one(pdf: pd.DataFrame) -> pd.DataFrame:
    return pdf + 1
