# Databricks notebook source
# MAGIC %md <i18n value="d6718279-32b1-490e-8a38-f1d6e3578184"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Entrenando en paralelo
# MAGIC
# MAGIC ¿Como podemos construir multiples modelos en paralelo? 
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) En esta lección:<br>
# MAGIC  - El uso de  <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandas.html" target="_blank"> **.groupBy().applyInPandas()** </a> para construir modelos en paralelo

# COMMAND ----------

# MAGIC %md <i18n value="35af29dc-0fc5-4e37-963d-3fbe86f4ba59"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Datos Dummy:
# MAGIC - **`device_id`**: 10 dispositivos diferentes
# MAGIC - **`record_id`**: 10k registros unicos
# MAGIC - **`feature_1`**: variable 1
# MAGIC - **`feature_2`**: variable 2
# MAGIC - **`feature_3`**: variable 3
# MAGIC - **`label`**: variable objetivo

# COMMAND ----------

import pyspark.sql.functions as f

df = (spark
      .range(1000*100)
      .select(f.col("id").alias("record_id"), (f.col("id")%10).alias("device_id"))
      .withColumn("feature_1", f.rand() * 1)
      .withColumn("feature_2", f.rand() * 2)
      .withColumn("feature_3", f.rand() * 3)
      .withColumn("label", (f.col("feature_1") + f.col("feature_2") + f.col("feature_3")) + f.rand())
     )

display(df)

# COMMAND ----------

# MAGIC %md <i18n value="b5f90a62-80fd-4173-adf0-6e73d0e31309"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Definimos el Esquema de retorno

# COMMAND ----------

train_return_schema = "device_id integer, n_used integer, model_path string, mse float"

# COMMAND ----------

# MAGIC %md <i18n value="e2ac315f-e950-48c6-9bb8-9ceede8f93dd"/>
# MAGIC
# MAGIC Definimos una funcuín que toma como input todos los datos vinculados a un dispositivo, entrena un modelo (usando mlflow) y devuelve el esquema definido.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:

    device_id = df_pandas["device_id"].iloc[0]
    n_used = df_pandas.shape[0]
    run_id = df_pandas["run_id"].iloc[0] # Pulls run ID to do a nested run

    
    X = df_pandas[["feature_1", "feature_2", "feature_3"]]
    y = df_pandas["label"]
    rf = RandomForestRegressor()
    rf.fit(X, y)

    
    predictions = rf.predict(X)
    mse = mean_squared_error(y, predictions) 

    
    with mlflow.start_run(run_id=run_id) as outer_run:
        experiment_id = outer_run.info.experiment_id
        print(f"Current experiment_id = {experiment_id}")

        
        with mlflow.start_run(run_name=str(device_id), nested=True, experiment_id=experiment_id) as run:
            mlflow.sklearn.log_model(rf, str(device_id))
            mlflow.log_metric("mse", mse)
            mlflow.set_tag("device", str(device_id))

            artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
            return_df = pd.DataFrame([[device_id, n_used, artifact_uri, mse]], 
                                    columns=["device_id", "n_used", "model_path", "mse"])

    return return_df 


# COMMAND ----------

# MAGIC %md <i18n value="2b6bf899-de7c-4ab9-b343-a11a832ddd77"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Aplicamos la función a datos agrupados dentro de un dataframe de Spark

# COMMAND ----------

with mlflow.start_run(run_name="Entrenando sobre todos los dispositivos") as run:
    run_id = run.info.run_id

    model_directories_df = (df
        .withColumn("run_id", f.lit(run_id))
        .groupby("device_id")
        .applyInPandas(train_model, schema=train_return_schema)
        .cache()
    )

combined_df = df.join(model_directories_df, on="device_id", how="left")
display(combined_df)

# COMMAND ----------

# MAGIC %md <i18n value="3f660cc6-4979-48dd-beea-9dab9b536230"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Predicción sobre los datos

# COMMAND ----------

apply_return_schema = "record_id integer, prediction float"

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    model_path = df_pandas["model_path"].iloc[0]

    input_columns = ["feature_1", "feature_2", "feature_3"]
    X = df_pandas[input_columns]

    model = mlflow.sklearn.load_model(model_path)
    prediction = model.predict(X)

    return_df = pd.DataFrame({
        "record_id": df_pandas["record_id"],
        "prediction": prediction
    })
    return return_df

prediction_df = combined_df.groupby("device_id").applyInPandas(apply_model, schema=apply_return_schema)
display(prediction_df)

# COMMAND ----------


