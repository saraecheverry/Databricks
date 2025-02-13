# Databricks notebook source
# MAGIC %md <i18n value="be8397b6-c087-4d7b-8302-5652eec27caf"/>
# MAGIC
# MAGIC
# MAGIC  
# MAGIC # Hyperopt
# MAGIC
# MAGIC Usemos HyperOpt para hacer busqueda de hiperparametros en modelos de una sola maquina.

# COMMAND ----------

paths_base = "mnt/testdatabricks/datasets_andercol_test/"

# COMMAND ----------

from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv(f"/{paths_base}/airbnb/sf-listings/airbnb-cleaned-mlflow.csv".replace("mnt/", "dbfs/mnt/")).drop(["zipcode"], axis=1)

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1),
                                                    df[["price"]].values.ravel(),
                                                    test_size = 0.2,
                                                    random_state = 42)

# COMMAND ----------

# MAGIC %md <i18n value="b84062c7-9fb2-4d34-a196-98e5074c7ad4"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC A continuación definimos la **función objetivo** donde construimos un <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" target="_blank">random forest's</a> y usamos como metrica el r2. 

# COMMAND ----------

# ANSWER
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score
from numpy import mean
  
def objective_function(params):
    max_depth = params["max_depth"]
    max_features = params["max_features"]

    regressor = RandomForestRegressor(max_depth=max_depth, max_features=max_features, random_state=42)

    r2 = mean(cross_val_score(regressor, X_train, y_train, cv=3))

    return -r2

# COMMAND ----------

# MAGIC %md <i18n value="7b10a96d-d868-4603-ab84-50388a8f50fc"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Definimos el espacio de busqueda para HyperOpt. Usamos *max_depth* y *max_features* como parametros.

# COMMAND ----------

from hyperopt import hp

max_features_choices =  ["auto", "sqrt", "log2"]
search_space = {
    "max_depth": hp.quniform("max_depth", 2, 10, 1),
    "max_features": hp.choice("max_features", max_features_choices)
}

# COMMAND ----------

# MAGIC %md <i18n value="6db6a36a-e1ca-400d-81fc-20ad5a794a01"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Para hacer la busqueda de hiperparametros cambiamos la clase *Trials* por *SparkTrials* que permite ejecutar la busqueda de hiperparametros en distintos ejecutores de Spark.
# MAGIC
# MAGIC *SparkTrials* tienen 3 parametros: 
# MAGIC - parallelism: El numero de modelos entrenados de forma **concurrente**
# MAGIC - timeout: Tiempo maximo de ejecución
# MAGIC - spark_session: La sesión de SPARK.
# MAGIC
# MAGIC

# COMMAND ----------

# ANSWER
from hyperopt import fmin, tpe, SparkTrials
import mlflow
import numpy as np


num_evals = 8
spark_trials = SparkTrials(parallelism=2)
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       trials=spark_trials,
                       max_evals=num_evals,
                       rstate=np.random.default_rng(42))


with mlflow.start_run(run_name="best_model"):
    best_max_depth = best_hyperparam["max_depth"]
    best_max_features = max_features_choices[best_hyperparam["max_features"]]

    regressor = RandomForestRegressor(max_depth=best_max_depth, max_features=best_max_features, random_state=42)
    regressor.fit(X_train, y_train)

    r2 = regressor.score(X_test, y_test)

    mlflow.log_param("max_depth", best_max_depth)
    mlflow.log_param("max_features", best_max_features)
    mlflow.log_metric("loss", r2)
