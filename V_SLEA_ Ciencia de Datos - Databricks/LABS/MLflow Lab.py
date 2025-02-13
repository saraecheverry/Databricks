# Databricks notebook source
# MAGIC %md <i18n value="2cf41655-1957-4aa5-a4f0-b9ec55ea213b"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Laboratorio
# MAGIC
# MAGIC Vamos a usar MLFlow dentro del ciclo de vida del modelamiento:
# MAGIC
# MAGIC 1. Cargamos el dataset y lo almacenamos en formato delta tanto para entrenamiento como para predicción. 
# MAGIC 1. Entrenamiento de un modelo mediante SparkML usando los componentes de MLflow: logging de parametros, metricas, etc. Incorporamos la versión de los datos.
# MAGIC 1. Registramos el modelo y pasamos al estado de "Staging" dentro de Registry.
# MAGIC 1. Agregamos una nueva columna *log_price* tanto a los datos de entrenamiento como a los de validación y actualizamos la tabla delta.
# MAGIC 1. Entrenamos un segundo modelo usando como variable objetivo *log_price* y repetimos el proceso sobre Mlflow 
# MAGIC 1. Comparamos los dos modelos a nivel de desempeño,
# MAGIC 1. Movemos el mejor modelo de Staging a PDN en el Registry. 
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) En este lab:<br>
# MAGIC - Uso de tablas delta.
# MAGIC - Uso de Mlflow

# COMMAND ----------

# MAGIC %md <i18n value="197ad07c-dead-4444-82de-67353d81dcb0"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ###  Paso 1. Creación de tablas delta

# COMMAND ----------

# MAGIC %md <i18n value="8e9e809d-4142-43d8-b361-830099a02d06"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Ya vimos que delta nos permite hacer versionamiento de datos, por tanto, tenemos trazabilidad y podemos volver a versiones anteriores de los datasets de entrenamiento. 

# COMMAND ----------

# %sql
# CREATE SCHEMA IF NOT EXISTS schema_slea

# COMMAND ----------

file_path = "dbfs:/user/hive/warehouse/schema_slea.db/listings"
airbnb_df = spark.read.format("delta").load(file_path)

display(airbnb_df)

# COMMAND ----------

from pyspark.sql.functions import col, sum

# Leer el DataFrame desde la ruta especificada
file_path = "dbfs:/user/hive/warehouse/schema_slea.db/listings"
airbnb_df = spark.read.format("delta").load(file_path)

# Crear una lista de expresiones que cuentan los valores nulos por columna
null_counts = [sum(col(column).isNull().cast("integer")).alias(column) for column in airbnb_df.columns]

# Aplicar las expresiones de agregación para contar los nulos
null_counts_df = airbnb_df.agg(*null_counts)

# Recoger los resultados en un DataFrame local
null_counts_local = null_counts_df.collect()[0].asDict()

# Filtrar columnas con menos de 150 nulos
columns_with_few_nulls = [column for column, count in null_counts_local.items() if count < 200]

# Filtrar el DataFrame para conservar solo las columnas con menos de 150 nulos
filtered_airbnb_df = airbnb_df.select(*columns_with_few_nulls)
filtered_airbnb_df = filtered_airbnb_df.dropna()
# Mostrar las primeras filas del DataFrame filtrado
display(filtered_airbnb_df)

# COMMAND ----------

from pyspark.sql.functions import regexp_extract, col

# Definir una expresión regular para extraer el número del string
regex_pattern = r"(\d+)"  # Busca una secuencia de uno o más dígitos

# Extraer el número del string
filtered_airbnb_df = filtered_airbnb_df.withColumn("price", regexp_extract(col("price"), regex_pattern, 1).cast("int"))

# COMMAND ----------

from pyspark.sql.types import NumericType
# Obtener las columnas numéricas
numeric_columns = [field.name for field in filtered_airbnb_df.schema.fields if isinstance(field.dataType, NumericType)]
# numeric_columns.append('price')
# Filtrar el DataFrame para conservar solo las columnas numéricas
numeric_airbnb_df = filtered_airbnb_df.select(*numeric_columns)

# Mostrar las primeras filas del DataFrame filtrado
display(numeric_airbnb_df)

# COMMAND ----------

# estudiante = "joaosoriolo"
# paths_base = "mnt/testdatabricks/datasets_andercol_test/"
working_dir = "user/hive/warehouse/schema_slea.db/"

# COMMAND ----------

# file_path = f"/{paths_base}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
# airbnb_df = spark.read.format("delta").load(file_path)

train_df, test_df = numeric_airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

train_delta_path = f"/{working_dir}/train.delta"
test_delta_path = f"/{working_dir}/test.delta"

dbutils.fs.rm(train_delta_path, True)
dbutils.fs.rm(test_delta_path, True)

train_df.write.mode("overwrite").format("delta").save(train_delta_path)
test_df.write.mode("overwrite").format("delta").save(test_delta_path)

# COMMAND ----------

# %fs ls "/user/hive/warehouse/schema_slea.db/"
display(dbutils.fs.ls(working_dir))

# COMMAND ----------

# MAGIC %md <i18n value="ead09bc6-c6f2-4dfa-bc9c-ddf41accc8f8"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC A continuación leamos las tablas especificando que queremos la primera versión. 
# MAGIC Ayuda: <a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">*HINT*</a>

# COMMAND ----------

# ANSWER
data_version = 0
train_df = spark.read.format("delta").option("versionAsOf", data_version).load(train_delta_path)
test_df = spark.read.format("delta").option("versionAsOf", data_version).load(test_delta_path)

# COMMAND ----------

# MAGIC %md <i18n value="2bf375c9-fb36-47a3-b973-82fa805e8b22"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Historia de la tabla Delta
# MAGIC Todas las operaciones sobre la tabla son almancenadas como metadata y pueden consultarse

# COMMAND ----------

history_table = spark.sql(f"DESCRIBE HISTORY delta.`{train_delta_path}`")
display(history_table)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS train_df;")
spark.sql(f"CREATE TABLE train_df USING DELTA LOCATION '{train_delta_path}'")

spark.sql("DROP TABLE IF EXISTS test_df;")
spark.sql(f"CREATE TABLE test_df USING DELTA LOCATION '{test_delta_path}'")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY train_df

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE EXTENDED train_df;

# COMMAND ----------

# MAGIC %md <i18n value="ffe159a7-e5dd-49fd-9059-e399237005a7"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Paso 2. Ejecución incicial sobre MLFlow
# MAGIC
# MAGIC Creamos el primer modelo logeando sobre MLFlow la siguiente metadata: versión de la tabla delta y su ubicación, parametros, y metricas. 
# MAGIC En este caso usemos *RFormula* con todas las covariables.  
# MAGIC
# MAGIC **HINT**:
# MAGIC ```{python}
# MAGIC from pyspark.ml.feature import RFormula
# MAGIC r_formula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip")
# MAGIC ```

# COMMAND ----------



# COMMAND ----------

# ANSWER
import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
# from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import RFormula
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

with mlflow.start_run(run_name="LR-All-Features") as run:
    # Pipeline
    r_formula = RFormula(formula="price ~ .", featuresCol="features", labelCol="label", handleInvalid="skip")
    lr = LinearRegression(labelCol="label", featuresCol="features")
    pipeline = Pipeline(stages=[r_formula, lr])
    pipeline_model = pipeline.fit(train_df)

    # Log Pipeline
    mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas())

    # Log parametros
    mlflow.log_param("version", data_version)
    mlflow.log_param("path", train_delta_path )
    # mlflow.log_param("label", train_df.select('price'))
    # mlflow.log_param("features", "all_features")

    # Metricas y preds
    pred_df = pipeline_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
    rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

    # Log metricas
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

# COMMAND ----------

# MAGIC %md <i18n value="3bac0fef-149d-4c25-9db1-94fbcd63ba13"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Paso 3. Registramos el modelo y lo movemos a *Staging*
# MAGIC
# MAGIC A continuación creamos movemos el modelo a Staging

# COMMAND ----------

# from mlflow.tracking.client import MlflowClient

# client = MlflowClient()

run_id = run.info.run_id
# estudiante = 'slea'
model_name = f"LinearRegression_mllib_lr"
model_uri = f"runs:/{run_id}/model"

# COMMAND ----------

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md <i18n value="78b33d27-0815-4d31-80a0-5e110aa96224"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Movemos el modelo a Staging

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# COMMAND ----------

# ANSWER
from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

# MAGIC %md <i18n value="b5f74e40-1806-46ab-9dd0-97b82d8f297e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Agreguemos una descripción al modelo registrado <a href="https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.update_registered_model" target="_blank">**HINT**</a>.

# COMMAND ----------

client.update_registered_model(
    name=model_details.name,
    description="Este modelo predice el precio de sitios de Airbnb."
)

# COMMAND ----------

# MAGIC %md <i18n value="03dff1c0-5c7b-473f-83ec-4a8283427280"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ###  Step 4. Ingenieria de datos...
# MAGIC
# MAGIC A continuación vamos a hacer algunas transformaciones sobre los datos para mejorar el modelo. En este caso, llevamos transformamos la variabe de *precio* a  *log_price*
# MAGIC Usamos delta para mantener las versiones anteriores del dataset 

# COMMAND ----------

from pyspark.sql.functions import col, log, exp
train_new = train_df.withColumn("log_price", log(col("price")))
test_new = test_df.withColumn("log_price", log(col("price")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## HINT
# MAGIC ### Evolución de esquema
# MAGIC Una de las funcionales más importantes de Delta y que es de mucha utilidad es [**Schema Evolution**](https://docs.delta.io/latest/delta-update.html#automatic-schema-evolution).  
# MAGIC Esta nos permite aplicar transformaciones sobre la tabla delta y conservarlas bajo ciertas condiciones (recordemos la inmutabilidad dentro de Spark) 
# MAGIC
# MAGIC Hay dos casos en los que podemos aplicar evolución de esquema:
# MAGIC 1. Agregado de nuevas columnas.
# MAGIC
# MAGIC Sin embargo, los cambios de esquema que requieran reescribir el parquet, **no están soportados**:
# MAGIC 1. Eliminación de una columna.
# MAGIC 2. Cambio de tipo de datos distintos a los mencionados.
# MAGIC 3. Cambio de nombres de columna que difieran sólo en mayúscula o minúscula

# COMMAND ----------

# MAGIC %md <i18n value="565313ed-2bca-4cc6-af87-1c0d509c0a69"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Para actualizar el dataframe escribimos sobre las rutas to *train_delta_path* y *test_delta_path*, usando la opción *mergeSchema*.. 
# MAGIC
# MAGIC Info sobre evolución de esquemas en delta: <a href="https://databricks.com/blog/2019/09/24/diving-into-delta-lake-schema-enforcement-evolution.html" target="_blank">blog</a> 

# COMMAND ----------

# ANSWER
# Update the write configuration to enable schema merge
train_new.write.format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .save(train_delta_path)

# COMMAND ----------

# Update the write configuration to enable schema merge
test_new.write.format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .save(test_delta_path)

# COMMAND ----------

history_table = spark.sql(f"DESCRIBE HISTORY delta.`{train_delta_path}`")
display(history_table)

# COMMAND ----------

# MAGIC %md <i18n value="735a36b6-7510-4e4f-9df8-4ae51f9f87dc"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Miremos la diferencia en los esquemas de los dataframes

# COMMAND ----------

set(train_new.schema.fields) ^ set(train_df.schema.fields)

# COMMAND ----------

# MAGIC %md <i18n value="0c7c986b-1346-4ff1-a4e2-ee190891a5bf"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Observemos de nuevo la historia de la tabla delta

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{train_delta_path}`"))

# COMMAND ----------

data_version = 1
train_delta_new = spark.read.format("delta").option("versionAsOf", data_version).load(train_delta_path)  
test_delta_new = spark.read.format("delta").option("versionAsOf", data_version).load(test_delta_path)

# COMMAND ----------

# MAGIC %md <i18n value="f29c99ca-b92c-4f74-8bf5-c74070a8cd50"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Paso 5. Usemos *log_price* como objetivo para un nuevo modelo 
# MAGIC Volvamos a seguir los pasos usados en el paso 2

# COMMAND ----------

with mlflow.start_run(run_name="lr_log_model") as run:
    # Log parameters
    mlflow.log_param("label", "log-price")
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("data_path", train_delta_path)    

    # Create pipeline
    r_formula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="label", handleInvalid="skip")  
    lr = LinearRegression(labelCol="label", predictionCol="log_prediction")
    pipeline = Pipeline(stages = [r_formula, lr])
    pipeline_model = pipeline.fit(train_delta_new)

    # Log model and update the registered model
    mlflow.spark.log_model(
        spark_model=pipeline_model,
        artifact_path="log-model",
        registered_model_name=model_name
    )  
    # mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas())
    mlflow.log_param("version", 1)
    mlflow.log_param("path", train_delta_path )

    # Create predictions and metrics
    pred_df = pipeline_model.transform(test_delta_new)
    # exp_df = pred_df.withColumn("prediction", exp(col("log_prediction")))
    exp_df = pred_df.withColumn("prediction", col("log_prediction"))
    rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(exp_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)  

    run_id = run.info.run_id

# COMMAND ----------



# COMMAND ----------

# MAGIC %md <i18n value="e5bd7bfb-f445-44b5-a272-c6ae2849ac9f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Step 6. Comparemos los modelos entrenados sobre distintas variaciones de los datos.
# MAGIC
# MAGIC Usamos MLflow's <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs" target="_blank">*mlflow.search_runs*</a> para identificar las ejecuciones usando como parametros de busqueda la versión de la tabla delta deseada.
# MAGIC
# MAGIC **HINT**
# MAGIC Filtrar usando los campos *params.data_path* y *params.data_version*

# COMMAND ----------

import mlflow

all_runs = mlflow.search_runs(search_all_experiments=True)
print(all_runs)

# COMMAND ----------

import mlflow

runs_model = mlflow.search_runs(
    filter_string="params.version = 0", search_all_experiments=True,
    order_by=["start_time DESC"], max_results=1,
)
display(runs_model)

# COMMAND ----------

# # ANSWER
# data_version = 0

# mlflow.search_runs(filter_string=f"params.data_path='{train_delta_path}' and params.data_version='{data_version}'")

# COMMAND ----------

import mlflow

runs_logmodel = mlflow.search_runs(
    filter_string="params.version = 1", search_all_experiments=True,
    order_by=["start_time DESC"], max_results=1,
)
display(runs_logmodel)

# COMMAND ----------

# MAGIC %md <i18n value="fd0fc3ae-7c2e-4d7d-90da-0b6e6b830496"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Cuales datos dan lugar al mejor modelo?

# COMMAND ----------

# MAGIC %md <i18n value="3056bfcc-7623-4410-8b1b-82cba24ae3dd"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Paso 7. Movemos el modelo con el mejor desempeño de Staging a PDN
# MAGIC
# MAGIC Modelos el modelo con el mejor desempeño a PDN.

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = [model_version_info.version for model_version_info in model_version_infos if model_version_info.run_id == runs_model['run_id'][0]][0]

# COMMAND ----------

new_model_version

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=new_model_version,
    description="Este modelo se construyo usando SparkML y usa como var target precio."
)

# COMMAND ----------

model_version_details = client.get_model_version(name=model_name, version=new_model_version)
model_version_details.status

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Finalmente, archivamos los modelos y los eliminamos

# COMMAND ----------

# client.transition_model_version_stage(
#     name=model_name,
#     version=new_model_version,
#     stage="Archived"
# )

# COMMAND ----------

for i in range(1,3):
    client.transition_model_version_stage(
        name=model_name,
        version=i,
        stage="Archived"
    )

# COMMAND ----------

client.delete_registered_model(model_name)
