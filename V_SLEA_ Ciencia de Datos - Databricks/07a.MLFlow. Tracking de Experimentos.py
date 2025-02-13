# Databricks notebook source
# MAGIC %md <i18n value="b27f81af-5fb6-4526-b531-e438c0fda55e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # MLflow
# MAGIC
# MAGIC Existen 3 problemas en el ciclo de vida del Modelamiento:
# MAGIC
# MAGIC * Es complicado mantener visibilidad y trazabilidad de los experimentos
# MAGIC * Es dificil garantizar la reproducibilidad de los experimentos
# MAGIC * No hay un estandar concreto respecto a la serialización de modelos. 
# MAGIC
# MAGIC
# MAGIC MLflow es un herramienta diseñada para facilitar estas dificultades 
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) En esta lección:<br>
# MAGIC * Usar Mlflow para vigilar experimentos, logear metricas y comparar ejecuciones.

# COMMAND ----------

# MAGIC %md <i18n value="b7c8a0e0-649e-4814-8310-ae6225a57489"/>
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="c1a29688-f50a-48cf-9163-ebcc381dfe38"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC De nuevo usaremos el dataset de Airbnb

# COMMAND ----------

paths_base = "mnt/testdatabricks/datasets_andercol_test/"
working_dir = "dbfs:/mnt/testdatabricks/datasets_andercol_test/airbnb"

# COMMAND ----------

file_path = f"/{paths_base}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------

# MAGIC %md <i18n value="9ab8c080-9012-4f38-8b01-3846c1531a80"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### MLflow Tracking
# MAGIC
# MAGIC Mlflow Tracking es la herramienta encargada de registrar los experimentos que realizamos. Funciona como una serie de decoradores y APIs que integramos dentro del código de entrenamiento (o incluso de los análisis exploratorios y de obtención de datos) para registrar, principalmente, tres tipos de objetos:
# MAGIC
# MAGIC
# MAGIC - Parámetros: inputs de modelos, rutas de archivos, etc. Son entradas de tipo clave-valor y suelen ser strings.
# MAGIC - Métricas: son valores numéricos que se obtienen de los modelos, como las métricas de evaluación, y que pueden actualizarse durante la ejecución de experimentos.
# MAGIC - Artefactos: son archivos blob en cualquier formato que se registran durante la ejecución de un experimento. Podemos registrar imágenes, los modelos serializados, datos en formato parquet, etc.
# MAGIC - Source: el codigo usado para ejecutar el experimento. 
# MAGIC
# MAGIC Tracking se articula al rededor del concepto de **runs**, que son ejecuciones del codigo. A su vez, las ejecuciones son agregadas dentro de experimentos.
# MAGIC
# MAGIC Podemos usar <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment" target="_blank">mlflow.set_experiment()</a> para definir un experimento, sin embargo, de no hacerlo se creará un experimento con el alcance vinculado a este notebook.

# COMMAND ----------

# MAGIC %md <i18n value="82786653-4926-4790-b867-c8ccb208b451"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Track Runs
# MAGIC
# MAGIC **NOTA**: Para los modelos construidos por Spark. Al día de hoy MlFlow solo puede logerar Pipelines

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

with mlflow.start_run(run_name="LR-Single-Feature") as run:
    # Pipeline
    vec_assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="price")
    pipeline = Pipeline(stages=[vec_assembler, lr])
    pipeline_model = pipeline.fit(train_df)

    # Log parametros
    mlflow.log_param("label", "price")
    mlflow.log_param("features", "bedrooms")

    # Log modelo
    mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas()) 

    # Evaluación
    pred_df = pipeline_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
    rmse = regression_evaluator.evaluate(pred_df)

    # Metricas
    mlflow.log_metric("rmse", rmse)

# COMMAND ----------

# MAGIC %md <i18n value="44bc7cac-de4a-47e7-bfff-6d2eb58172cd"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Ejecutamos de nuevo el codigo pero esta vez usando todas las variables.

# COMMAND ----------

from pyspark.ml.feature import RFormula

with mlflow.start_run(run_name="LR-All-Features") as run:
    # Pipeline
    r_formula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip")
    lr = LinearRegression(labelCol="price", featuresCol="features")
    pipeline = Pipeline(stages=[r_formula, lr])
    pipeline_model = pipeline.fit(train_df)

    # Log Pipeline
    mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas())

    # Log parametros
    mlflow.log_param("label", "price")
    mlflow.log_param("features", "all_features")

    # Metricas y preds
    pred_df = pipeline_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
    rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

    # Log metricas
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

# COMMAND ----------

# MAGIC %md <i18n value="70188282-8d26-427d-b374-954e9a058000"/>
# MAGIC
# MAGIC
# MAGIC Por ultimo, combinemos el codigo de entrenamiento con alguna transformación y agregamos un artefacto de tipo imagen al log.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, log, exp
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature

with mlflow.start_run(run_name="LR-Log-Price") as run:
    # Take log of price
    log_train_df = train_df.withColumn("log_price", log(col("price")))
    log_test_df = test_df.withColumn("log_price", log(col("price")))

    # Log parameter
    mlflow.log_param("label", "log_price")
    mlflow.log_param("features", "all_features")

    # Create pipeline
    r_formula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="log_price", handleInvalid="skip")  
    lr = LinearRegression(labelCol="log_price", predictionCol="log_prediction")
    pipeline = Pipeline(stages=[r_formula, lr])
    pipeline_model = pipeline.fit(log_train_df)

    # Make predictions
    pred_df = pipeline_model.transform(log_test_df)
    exp_df = pred_df.withColumn("prediction", exp(col("log_prediction")))

    # Evaluate predictions
    rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(exp_df)
    
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Log Model
    mlflow.spark.log_model(pipeline_model, "log-model", input_example=log_train_df.limit(5).toPandas())
    # Log artifact
    plt.clf()

    log_train_df.toPandas().hist(column="log_price", bins=100)
    fig = plt.gcf()
    mlflow.log_figure(fig, f"log_normal.png")
    plt.show()

# COMMAND ----------

pipeline_model.stages[1].summary.explainedVariance

# COMMAND ----------

# MAGIC %md <i18n value="66785d5e-e1a7-4896-a8a9-5bfcd18acc5c"/>
# MAGIC
# MAGIC A continuación, podemos ver como MLFlow nos permite mantener trazabilidad ya que podemos volver sobre las ejecuciones anteriores y comparar el desempeño de los modelos.  
# MAGIC Esto lo podemos hacer tanto desde el UI de MLFlow como desde la la API.
# MAGIC

# COMMAND ----------

# MAGIC %md <i18n value="0b1a68e1-bd5d-4f78-a452-90c7ebcdef39"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Accediendo a ejecuciones pasadas 
# MAGIC
# MAGIC Podemos acceder a ejecuciones pasadas desde el API de Python de forma programatica. Lo hacemos a través del cliente de MLFlow:  **`MlflowClient`** 

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

client.list_experiments()

# COMMAND ----------

client.list_experiments()[1].experiment_id

# COMMAND ----------

# MAGIC %md <i18n value="dcd771b2-d4ed-4e9c-81e5-5a3f8380981f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC You can also use <a href="https://mlflow.org/docs/latest/search-syntax.html" target="_blank">search_runs</a> to find all runs for a given experiment.

# COMMAND ----------

experiment_id = client.list_experiments()[1].experiment_id
runs_df = mlflow.search_runs(experiment_id)

display(runs_df)

# COMMAND ----------

# MAGIC %md <i18n value="68990866-b084-40c1-beee-5c747a36b918"/>
# MAGIC
# MAGIC Observemos las metricas de la última ejecución

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
runs[0].data.metrics

# COMMAND ----------

runs[0].info.run_id

# COMMAND ----------

# MAGIC %md <i18n value="63ca7584-2a86-421b-a57e-13d48db8a75d"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Consumo de modelos
# MAGIC
# MAGIC Podemos acceder a los modelos serializados mediante la integración de  MLFlow con <a href="https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html" target="_blank">Spark</a>

# COMMAND ----------

model_path = f"runs:/{run.info.run_id}/log-model"
loaded_model = mlflow.spark.load_model(model_path)

display(loaded_model.transform(test_df))
