# Databricks notebook source
# MAGIC %md <i18n value="2630af5a-38e6-482e-87f1-1a1633438bb6"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # AutoML
# MAGIC
# MAGIC <a href="https://docs.databricks.com/applications/machine-learning/automl.html" target="_blank">Databricks AutoML</a> nos ayuda a acelerar el proceso de construcción de modelos. Se encargar de hacer un analisis exploratorio sobre los datos y preprocesamiento y luego define una serie de experimentos donde construye, tunea y evalua distintos modelos (con el uso de HyperOpt). 
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) En esta lección:<br>
# MAGIC  - Uso de AutoML para crear y tunear modelos de forma automatica. 

# COMMAND ----------

# MAGIC %md <i18n value="7aa84cf3-1b6c-4ba4-9249-00359ee8d70a"/>
# MAGIC
# MAGIC
# MAGIC Al día de hoy, AutoML se vale de una combinación de XGBoost y Sklearn (modelos de una sola maquina).

# COMMAND ----------

paths_base = "mnt/testdatabricks/datasets_andercol_test/"

# COMMAND ----------

file_path = f"/{paths_base}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="1b5c8a94-3ac2-4977-bfe4-51a97d83ebd9"/>
# MAGIC
# MAGIC El uso de AutoML tiene los siguientes parametros: 
# MAGIC
# MAGIC - dataset: Un dataframe de Spark o Pandas que contiene las variables y el target. AutoML usa modelos de una maquina por lo que convierte SparkDF a Pandas (Cuidado OOM!!)
# MAGIC - target_col: Columna objetivo 
# MAGIC
# MAGIC Parametros adicionales
# MAGIC - primary_metric: Metrica principal de comparación para definir el mejor modelo.
# MAGIC - timeout_minutes: Tiempo maximo para la ejecución de los modelos. 
# MAGIC - max_trials:  Numero maximo de ejecuciones. 

# COMMAND ----------

from databricks import automl

summary = automl.regress(train_df, target_col="price", primary_metric="rmse", timeout_minutes=5, max_trials=10)

# COMMAND ----------

# MAGIC %md <i18n value="57d884c6-2099-4f34-b840-a4e873308ffe"/>
# MAGIC
# MAGIC
# MAGIC  
# MAGIC
# MAGIC AutoML produce dos notebooks y un experimento registrado en MLFLow:
# MAGIC - Data exploration notebook: Notebook de analisis exploratorio de las variables: frecuencias, etc. 
# MAGIC - Best trial notebook: Notebook del codigo fuente para reproducir el mejor modelo encontrado. 
# MAGIC - MLflow experiment: Información de las ejecuciones: metricas, artefactos, parametros, etc. 
# MAGIC

# COMMAND ----------

print(summary.best_trial)

# COMMAND ----------

# MAGIC %md <i18n value="3c0cd1ec-8965-4af3-896d-c30938033abf"/>
# MAGIC
# MAGIC Podemos consumir el modelo

# COMMAND ----------


import mlflow

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = test_df.withColumn("prediction", predict(*test_df.drop("price").columns))
display(pred_df)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE on test dataset: {rmse:.3f}")
