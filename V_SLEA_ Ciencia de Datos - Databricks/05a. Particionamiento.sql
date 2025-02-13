-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Particiones
-- MAGIC El [particionamiento de tablas](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-PartitionedTables) es una técnica heredada del motor Apache Hive, desarrollado en su momento para ejecutar consultas SQL en Apache Hadoop. Cuando particionamos una tabla, forzamos a que los datos se almacenen en distintos directorios dentro del servicio de almacenamiento que utilicemos (HDFS con Hive, AWS S3 para nuestro caso), esto, basado en los valores de una o más columnas que elijamos. De esta manera, cuando con una consulta, queremos seleccionar un subconjunto de datos a partir de un determinado filtro que coincide con el criterio de particionamiento, el motor de procesamiento, en nuestro caso Apache Spark, sólo tiene que buscar dentro de la partición correspondiente, haciendo que la consulta sea mucho más rápida que si la tabla no estuviera particionada.  
-- MAGIC Por otra parte, el particionamiento tiene dos grandes desventajas:
-- MAGIC 1. Si elegimos una columna de alta cardinalidad, es decir con muchos valores posibles, el particionamiento pasa a ser un problema porque al tener una gran cantidad de particiones, se generan muchas operaciones de lectura que ralentizan las consultas.
-- MAGIC 2. Si la columna elegida para particionar al momento del diseño de la tabla, luego no es usada en consultas, estamos penalizando las mismas sin necesidad.
-- MAGIC
-- MAGIC Para entender mejor el paricionamiento, usemos la tabla top_3_productos creada en el módulo anterior.
-- MAGIC
-- MAGIC **Nota:** es importante diferenciar este particionamiento del que realiza Apache Spark en memoria al procesar los datos para poder paralelizar el procesamiento. Este está fuera del alcance de este curso, pero pueden ver más detalles en este [link](https://spark.apache.org/docs/latest/rdd-programming-guide.html#parallelized-collections)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Estructura de directorios
-- MAGIC Primero observemos cómo se almacenaron los archivos parquet, recordando que las tablas internas tienen un directorio predefinido establecido por Databricks.

-- COMMAND ----------

-- MAGIC %fs ls "/user/hive/warehouse/datalytics_databricks.db/top_3_productos/"

-- COMMAND ----------

-- MAGIC %fs ls "/user/hive/warehouse/datalytics_databricks.db/"

-- COMMAND ----------

-- MAGIC %python
-- MAGIC %fs ls "/user/hive/warehouse/datalytics_databricks.db/"

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Ahora, suponiendo que la mayoría de las consultas se harán consultando un mes en particular, creamos una nueva tabla igual a la que tenemos pero particionada por *anio_mes_factura*.

-- COMMAND ----------

-- DROP TABLE datalytics_databricks.top_3_productos_anio_mes;
CREATE TABLE datalytics_databricks.top_3_productos_anio_mes
USING PARQUET
PARTITIONED BY (anio_mes_factura)
AS
SELECT * FROM datalytics_databricks.top_3_productos

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Volvemos a consultar la estructura de directorios, vemos que para la tabla particionada tenemos un directorio por mes con el formato **anio_mes_factura=valor_año_mes**

-- COMMAND ----------

-- MAGIC %fs ls "/user/hive/warehouse/datalytics_databricks.db/top_3_productos_anio_mes/"

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Ahora haremos una consulta simple filtrando para un mes en particular sobre ambas tablas y comparamos el plan de ejecución. En el mismo observamos que para la tabla particionada, Spark usa la opción de **PartitionFilters** para optimizar la búsqueda de los datos.   
-- MAGIC **Nota:** el particionamiento realizado es a modo de demostración y explicación, como estamos trabajando con poco volumen de datos, no vamos a poder advertir los beneficios del mismo en los tiempos de respuesta. Incluso sería mejor no particionar la tabla.

-- COMMAND ----------

EXPLAIN
SELECT * 
FROM datalytics_databricks.top_3_productos
WHERE anio_mes_factura=201111

-- COMMAND ----------

EXPLAIN
SELECT * 
FROM datalytics_databricks.top_3_productos_anio_mes
WHERE anio_mes_factura=201111
