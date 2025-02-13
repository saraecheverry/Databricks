# Databricks notebook source
# MAGIC %md <i18n value="04aa5a94-e0d3-4bec-a9b5-a0590c33a257"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Model Registry
# MAGIC
# MAGIC Model Registry es la plataforma colaborativa incoporada en MLflow que permite compartir y distribuir modelos entre las etapas de pruebas y producción. Mediante principios de gobierno, visibilidad y desempeño. 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) En esta lección :<br>
# MAGIC  - Registro de modelos usando Mlflow
# MAGIC  - Manejo del ciclo de vida de Modelos
# MAGIC  - Archivo de Modelos
# MAGIC  

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="5802ff47-58b5-4789-973d-2fb855bf347a"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### MlFlow Model Registry
# MAGIC
# MAGIC El Model Registry es un repositorio centralizado de modelos, una colección de APIS, y una UI pensadas para la administración del ciclo de vida de un modelo. Provee linage (que experimento y que ejecución y que datos produjo el modelo), versionamiento de modelos, transición entre etapas (staging a pdn), comentarios y opciones de despliegue.
# MAGIC
# MAGIC Registry tiene las siguientes caracteristicas:<br><br>
# MAGIC - Repositorio central: Repositorio de almacenamiento de modelos. Un modelo registrado posee un nombre unico, una version, un estado y metadata adicional. 
# MAGIC - Versionamiento de Modelos: Registro automatico de las nuevas versiones cuando los modelos son actualizados.
# MAGIC - Manejo de estados: Tag que permiten definir el ciclo de vida de los modelos.
# MAGIC - Integración con CI/CD: Administración de transición de estados, reviews y aprovaciones embebidas en los ciclos de CI/CD lo que refuerza el gobierno y el control.  
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="7f34f7da-b5d2-42af-b24d-54e1730db95f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Registro de modelos
# MAGIC
# MAGIC Vamos a registrar un modelo de forma programatica. Lo mismo puede lograrse del UI de MlFlow. 

# COMMAND ----------

# MAGIC %md <i18n value="cbc59424-e45b-4179-a586-8c14a66a61a1"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Vamos a entrenar un modelo mediante y vincularlo a MLflow con <a href="https://docs.databricks.com/applications/mlflow/databricks-autologging.html" target="_blank">autologging</a>. Para algunos modelos, MLFlow a diseñado  autologging que permite logear metricas, parametros y artefactos sin la necesidad de sentencias de codigo especificas.
# MAGIC
# MAGIC Podemos usar autologin de distintas maneras: 
# MAGIC
# MAGIC   1. Llamar **`mlflow.autolog()`** antes del codigo de entrenamiento. Esto habilita autologging para las librerias de ML que ya son soportadas en tanto sean importadas.
# MAGIC   2. Habilitad autologging a nivel de la consola de administración del Workspace de Databricks.
# MAGIC   3. USar los modulos especificos de autologin para las librerias compatibles (e.g. **`mlflow.spark.autolog()`**)
# MAGIC

# COMMAND ----------

paths_base = "mnt/testdatabricks/datasets_andercol_test/"

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv(f"/{paths_base}airbnb/sf-listings/airbnb-cleaned-mlflow.csv".replace("mnt", "/dbfs/mnt"))
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

with mlflow.start_run(run_name="LR Model") as run:
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, log_models=True)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    signature = infer_signature(X_train, lr.predict(X_train))

# COMMAND ----------

# MAGIC %md <i18n value="1322cac5-9638-4cc9-b050-3545958f3936"/>
# MAGIC
# MAGIC Creamos un nombre unico

# COMMAND ----------

model_name = f"joaosoriolo_sklearn_lr"
model_name

# COMMAND ----------

# MAGIC %md <i18n value="0777e3f5-ba7c-41c4-a477-9f0a5a809664"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finalmente, registramos el modelo.

# COMMAND ----------

run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="22756858-ff7f-4392-826f-f401a81230c4"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC  **Abrimos la pestaña *Models* en la izquierda de la pantalla y podemos observar el modelo registrado**:
# MAGIC
# MAGIC * Vemos quien entreno el modelo y que codigo fue usado. 
# MAGIC * Vemos una historia de las acciones que se han aplicado a este modelo. 
# MAGIC * Vemos la versión del modelo. 
# MAGIC
# MAGIC
# MAGIC ## Metadata 
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://lh3.googleusercontent.com/d/1oJ0gAoIi8X25TMJ-APB84YwlJBJQMiTI" width="1200"/>
# MAGIC </div>
# MAGIC
# MAGIC ## Linaje
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://lh3.googleusercontent.com/d/1kB0iIKLxbyLUcb6M9SDHUURuuNXuk6w4" width="1200"/>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md <i18n value="481cba23-661f-4de7-a1d8-06b6be8c57d3"/>
# MAGIC
# MAGIC
# MAGIC Miremos el status

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

# MAGIC %md <i18n value="10556266-2903-4afc-8af9-3213d244aa21"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Agreguemos una nueva descripción

# COMMAND ----------

client.update_registered_model(
    name=model_details.name,
    description="Este modelo predice el precio de sitios de Airbnb."
)

# COMMAND ----------

# MAGIC %md <i18n value="5abeafb2-fd60-4b0d-bf52-79320c10d402"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Add a version-specific description.

# COMMAND ----------

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="Este modelos se construyó usando OLS."
)

# COMMAND ----------

# MAGIC %md <i18n value="aaac467f-3a52-4428-a119-8286cb0ac158"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Despliegue
# MAGIC
# MAGIC Model Registry define distintas etapas de transición: **None**, **Staging**, **Production**, y **Archived**. Cada etapa posee un significado unico. Por ejemplo, **Staging** es para modelos que siguen en pruebas en tanto, **Production** es para modelos que han pasado el proceso de validación y van a ser puestos en producción.  
# MAGIC
# MAGIC Se pueden asignar distintos permisos a los usuarios para gobernar las transiciones

# COMMAND ----------

# MAGIC %md <i18n value="dff93671-f891-4779-9e41-a0960739516f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Vamos a pasar el modelo a Producción

# COMMAND ----------

import time

time.sleep(10) 

# COMMAND ----------

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md <i18n value="4dc7e8b7-da38-4ce1-a238-39cad74d97c5"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Miremos el nuevo estado

# COMMAND ----------

client.get_registered_model(model_name)

# COMMAND ----------

model_version_details = client.get_model_version(
    name=model_details.name,
    version=model_details.version
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

# MAGIC %md <i18n value="ba563293-bb74-4318-9618-a1dcf86ec7a3"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Obtengamos el modelo mediante **`pyfunc`**.  

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# MAGIC %md <i18n value="e1bb8ae5-6cf3-42c2-aebd-bde925a9ef30"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Usemos el modelo

# COMMAND ----------

model_version_1.predict(X_test)

# COMMAND ----------

# MAGIC %md <i18n value="75a9c277-0115-4cef-b4aa-dd69a0a5d8a0"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Desplegando una nueva versión
# MAGIC
# MAGIC Podemos crear multiples versiones de un modelos registrado. De esta forma, podemos crear un ciclo logico de evolución mediante las transiciones de staging y pdn. 

# COMMAND ----------

# MAGIC %md <i18n value="2ef7acd0-422a-4449-ad27-3a26f217ab15"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Podemos crear la nueva versión desde el momento que logeamos el modelo. 

# COMMAND ----------

from sklearn.linear_model import Ridge

with mlflow.start_run(run_name="LR Ridge Model") as run:
    alpha = .9
    ridge_regression = Ridge(alpha=alpha)
    ridge_regression.fit(X_train, y_train)


    mlflow.sklearn.log_model(
        sk_model=ridge_regression,
        artifact_path="sklearn-ridge-model",
        registered_model_name=model_name,
    )

    mlflow.log_params(ridge_regression.get_params())
    mlflow.log_metric("mse", mean_squared_error(y_test, ridge_regression.predict(X_test)))

# COMMAND ----------

client.get_registered_model(model_name)

# COMMAND ----------

# MAGIC %md <i18n value="dc1dd6b4-9e9e-45be-93c4-5500a10191ed"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Ponemos el nuevo modelo en Staging. 

# COMMAND ----------

import time

time.sleep(10)

client.transition_model_version_stage(
    name=model_details.name,
    version=2,
    stage="Staging"
)

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="fe857eeb-6119-4927-ad79-77eaa7bffe3a"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Miremos en el UI:
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://lh3.googleusercontent.com/d/1M0uIY5rdBvK8Fh5gtj-WieOfsAx9nfPe" width="1200"/>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="6f568dd2-0413-4b78-baf6-23debb8a5118"/>
# MAGIC
# MAGIC
# MAGIC Igualmente podemos buscar la nueva versión desde el cliente de MlFlow. 

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

# MAGIC %md <i18n value="4fb5d7c9-b0c0-49d5-a313-ac95da7e0f91"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Agregamos una descripción

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=new_model_version,
    description=f"Este modelo esta construido mediante Ridge con un alpha: {alpha}"
)

# COMMAND ----------

# MAGIC %md <i18n value="10adff21-8116-4a01-a309-ce5a7d233fcf"/>
# MAGIC
# MAGIC
# MAGIC Podemos pasar el modelo a producción y archivar las versiones anteriores: 

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production", 
    archive_existing_versions=True # Archieve existing model in production 
)

# COMMAND ----------

# MAGIC %md <i18n value="e3caaf08-a721-425b-8765-050c757d1d2e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Borramos la version 1.  
# MAGIC
# MAGIC **NOTA**: No se puede borrar un modelo sin antes archivarlo. 

# COMMAND ----------

client.delete_model_version(
    name=model_name,
    version=1
)

# COMMAND ----------

# MAGIC %md <i18n value="a896f3e5-d83c-4328-821f-a67d60699f0e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Archivamos la version 2. 

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Archived"
)

# COMMAND ----------

# MAGIC %md <i18n value="0eb4929d-648b-4ae6-bca3-aff8af50f15f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Borramos el modelo registrado. 

# COMMAND ----------

client.delete_registered_model(model_name)

# COMMAND ----------


