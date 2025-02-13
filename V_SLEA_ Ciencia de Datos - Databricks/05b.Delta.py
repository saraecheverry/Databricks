# Databricks notebook source
# MAGIC %md <i18n value="fd2d84ac-6a17-44c2-bb92-18b0c7fef797"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Delta Review
# MAGIC
# MAGIC Operaciones basicas para hacer sentido de Delta
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) En este lección:<br>
# MAGIC - Creación de Delta
# MAGIC - Lectura de tablas Delta
# MAGIC - Actualización de tablas Delta
# MAGIC - Acceso a versiones anteriores mediante *time travel*</a>
# MAGIC - <a href="https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html" target="_blank">Entendimiento del log transaccional</a>
# MAGIC
# MAGIC Usaremos los datos de San Francisco Airbnb de http://insideairbnb.com/get-the-data.html

# COMMAND ----------

# MAGIC %md <i18n value="68fcecd4-2280-411c-94c1-3e111683c6a3"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ###Por qué Delta?<br><br>
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://lh3.googleusercontent.com/d/1R9ghJk5f2nnGcQJ2vxCuJ6QLe7VvD-w2" width="500"/>
# MAGIC </div>
# MAGIC <div> 
# MAGIC
# MAGIC </div>
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://lh3.googleusercontent.com/d/1uRcxr-i_f2ygdvESosSdibc15TLiJS34" width="500"/>
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC Delta Lake es un formato open source de almacenamiento que combina **confianza y desempeño** a los data lakes. Para esto, Delta habilita transacciones ACID, manejo escalable de metada y unicidad de operaciones en batch y en streaming sobre un formato de almacenamiento escalable como Parquet.  
# MAGIC
# MAGIC Delta Lake se habilita sobre los lagos de datos ya existentes y es compatible de manera nativa con Spark. 

# COMMAND ----------

# MAGIC %md <i18n value="8ce92b68-6e6c-4fd0-8d3c-a57f27e5bdd9"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ###Creando una tabla delta
# MAGIC En primer lugar leemos los datos como un Dataframe

# COMMAND ----------

working_dir = "dbfs:/mnt/testdatabricks/datasets_andercol_test/airbnb/sf-listings/sf-listings_delta/"
cleaned_username = "joaosoriolo"

# COMMAND ----------

file_path = "dbfs:/mnt/testdatabricks/datasets_andercol_test/airbnb/sf-listings/"
dbutils.fs.ls(file_path)

# COMMAND ----------

file_path = f"dbfs:/mnt/testdatabricks/datasets_andercol_test/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/"
airbnb_df = spark.read.format("parquet").load(file_path)

display(airbnb_df)

# COMMAND ----------

# MAGIC %md <i18n value="c100b529-ac6b-4540-a3ff-4afa63577eee"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Podemos reescribir la tabla en un formato delta con la siguiente instrucción

# COMMAND ----------

# SparkDF a Delta
dbutils.fs.rm(working_dir, True)
airbnb_df.write.format("delta").mode("overwrite").save(working_dir)

# COMMAND ----------

# MAGIC %md <i18n value="090a31f6-1082-44cf-8e2a-6c659ea796ea"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Podemos registrar una tabla delta en el metastore

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {cleaned_username}")
spark.sql(f"USE {cleaned_username}")

airbnb_df.write.format("delta").mode("overwrite").saveAsTable("delta_review")

# COMMAND ----------

# MAGIC %md <i18n value="732577c2-095d-4278-8466-74e494a9c1bd"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Delta soporta particionamiento. El particionamiento coloca los datos con el mismo valor de la columna de particion en el mismo directorio.Así, las operaciones con una operación de filtro sobre la columna de partición solo leerá los directorios compatibles con el filtro. Esta optimización es lo que se conoce como **Partition pruning**. 

# COMMAND ----------

airbnb_df.write.format("delta").mode("overwrite").partitionBy("neighbourhood_cleansed").option("overwriteSchema", "true").save(working_dir)

# COMMAND ----------

# MAGIC %md <i18n value="e9ce863b-5761-4676-ae0b-95f3f5f027f6"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ###Entendiendo el <a href="https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html" target="_blank"> log transaccional </a>
# MAGIC Podemos observar como delta almacena los distintos barrios en directorios separados. Adicionalmente, vemos un directorio llamado _delta_log.

# COMMAND ----------

display(dbutils.fs.ls(working_dir))

# COMMAND ----------

# MAGIC %md <i18n value="ac970bba-1cf6-4aa3-91bb-74a797496eef"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://lh3.googleusercontent.com/d/10ftlZsgTFYkph0vtFo2XBh8Ilf1yVN1O"/>
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC Cuando se crea una tabla delta se crea automaticamente el log transaccional en el subdirectorio *_delta_log*. A medida que la tabla sufre transformaciones los cambios son almacenados de forma ordenada y en "commits" atomicos en el log transaccional. Cada commit es un archivo json. Cambios sobre la tabla generan nuevos json.

# COMMAND ----------

display(dbutils.fs.ls(f"{working_dir}/_delta_log/"))

# COMMAND ----------

# MAGIC %md <i18n value="2905b874-373b-493d-9084-8ff4f7583ccc"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Miremos un log transaccional
# MAGIC
# MAGIC Las cuatro columnas representan distintas partes del primer commit a la tabla delta 
# MAGIC - La columna **add** continene estadisticas sobre el dataframe en general y sobre columnas individuales. 
# MAGIC - La columna **commitInfo** tienen información util sobre que tipo de operación se hizo (WRITE o READ) y quien la hizo. 
# MAGIC - La columna **metaData** tiene información sobre el esquema de los datos.
# MAGIC - LA columna **protocol** informa la versión minima necesaria de Delta para leer o escribir sobre esta tabla delta. 

# COMMAND ----------

display(spark.read.json(f"{working_dir}/_delta_log/00000000000000000000.json"))

# COMMAND ----------

display(spark.read.json(f"{working_dir}/_delta_log/00000000000000000001.json"))

# COMMAND ----------

display(dbutils.fs.ls(f"{working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md <i18n value="9f817cd0-87ec-457b-8776-3fc275521868"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Lectura de una tabla delta

# COMMAND ----------

df = spark.read.format("delta").load(working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="faba817b-7cbf-49d4-a32c-36a40f582021"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Actualizando una tabla delta
# MAGIC
# MAGIC Filtramos donde el host es un superhost

# COMMAND ----------

df_update = airbnb_df.filter(airbnb_df["host_is_superhost"] == "t")
display(df_update)

# COMMAND ----------

df_update.write.format("delta").mode("overwrite").save(working_dir)

# COMMAND ----------

df = spark.read.format("delta").load(working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="e4cafdf4-a346-4729-81a6-fdea70f4929a"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Archivos en la particion Bayview luego de la actualización. Recordemos, cada archivo en el directorio es un snapshots del dataframe que se puede vincular a commits diferentes en el log transaccional. 

# COMMAND ----------

display(dbutils.fs.ls(f"{working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md <i18n value="25ca7489-8077-4b23-96af-8d801982367c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC #Delta Time Travel

# COMMAND ----------

# MAGIC %md
# MAGIC La próxima funcionalidad que veremos es [Time Travel](https://docs.delta.io/latest/delta-batch.html#-deltatimetravel). 
# MAGIC Como al realizar una operación sobre una tabla Delta, automáticamente se guarda la historia de la misma mediante [versionado](https://docs.databricks.com/delta/versioning.html), es posible consultar una tabla, ya sea en estructura o en datos, en un determinado punto anterior en el tiempo. De esta manera podemos implementar casos de uso como:
# MAGIC - Recrear análisis, reportes o resultados de un modelo de machine learning. Esto puede ser útil para *debugging* o  auditoría, especialmente cuando existan **regulaciones** en torno al manejo de datos.
# MAGIC - Corregir errores en los datos.
# MAGIC - Proveer aislamiento al consultar datos en tablas que cambian constantemente.
# MAGIC - Escribir consultas temporales complejas.
# MAGIC                                                       
# MAGIC Por ejemplo, en nuestro caso, sobreescribir el dataframe por completo... y si necesitamos una versión anterior? 

# COMMAND ----------

working_dir

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS train_delta;")
spark.sql(f"CREATE TABLE train_delta USING DELTA LOCATION '{working_dir}'")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY train_delta

# COMMAND ----------

# MAGIC %md <i18n value="61faa23f-d940-479c-95fe-5aba72c29ddf"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Mediante el uso de **`versionAsOf`** podemos volver facilmente a una versión anterior de la tabla.

# COMMAND ----------

df = spark.read.format("delta").option("versionAsOf", 0).load(working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="5664be65-8fd2-4746-8065-35ee8b563797"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC De igual forma, podemos volver a versiones anteriores con el uso del timestamp

# COMMAND ----------

time_stamp_string = str(spark.sql("DESCRIBE HISTORY train_delta").collect()[-1]["timestamp"])

df = spark.read.format("delta").option("timestampAsOf", time_stamp_string).load(working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="6cbe5204-fe27-438a-af54-87492c2563b5"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Cuando estamos satisfechos con una tabla podemos limpiar los directorios con el uso de **`VACUUM`**. Vacuum toma como parametro un periodo de retención en horas.

# COMMAND ----------

# MAGIC %md <i18n value="4da7827c-b312-4b66-8466-f0245f3787f4"/>

# COMMAND ----------

# MAGIC %sql
# MAGIC SET spark.databricks.delta.retentionDurationCheck.enabled = false

# COMMAND ----------

from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, working_dir)
delta_table.vacuum(0)

# COMMAND ----------

# MAGIC %md <i18n value="b845b2ea-2c11-4d6e-b083-d5908b65d313"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Como usamos vacuum con un periodo de retención de 0 horas. La unica información que veremos en los directorios es aquella vinculada al último commit. 

# COMMAND ----------

display(dbutils.fs.ls(f"{working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md <i18n value="a7bcdad3-affb-4b00-b791-07c14f5e59d5"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Vacuum elimina los archivos referenciados por la tabla delta. Por tanto, ya no podemos usar time travel para volver a versiones anteriores. 

# COMMAND ----------

df = spark.read.format("delta").option("versionAsOf", 0).load(working_dir)
display(df)
