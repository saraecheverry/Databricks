# Databricks notebook source
# MAGIC %md <i18n value="1fa7a9c8-3dad-454e-b7ac-555020a4bda8"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Hyperopt
# MAGIC
# MAGIC
# MAGIC Hyperopt es una libreria de python para "optimización serial y paralela sobre espacios de busqueda complejos, los cuales se componen de valores reales y discretos"
# MAGIC
# MAGIC En un ciclo de entrenamiento Hyperopt se usa para acelerar el proceso de busqueda de hiperparametros mediante metodos informados que son más eficientes que los tradicionales. 
# MAGIC
# MAGIC Hay dos formas de usar Hyperopt con Spark:
# MAGIC * Hyperopt sobre una sola maquina con un modelo distribuido (e.g. SparkML)
# MAGIC * Hyperopt distribuido sobre modelos que usan una sola maquina (e.g. scikit-learn). 

# COMMAND ----------

# MAGIC %md <i18n value="2340cdf4-9753-41b4-a613-043b90f0f472"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Nuevamente usaremos el dataset de Airbnb

# COMMAND ----------

paths_base = "mnt/testdatabricks/datasets_andercol_test/"
working_dir = "dbfs:/mnt/testdatabricks/datasets_andercol_test/airbnb"

# COMMAND ----------

file_path = f"/{paths_base}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, val_df, test_df = airbnb_df.randomSplit([.6, .2, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="37bbd5bd-f330-4d02-8af6-1b185612cdf8"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Creamos el pipeline de entrenamiento y estimador de evaluación

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

rf = RandomForestRegressor(labelCol="price", maxBins=40, seed=42)
pipeline = Pipeline(stages=[string_indexer, vec_assembler, rf])
regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price")

# COMMAND ----------

# MAGIC %md <i18n value="e4627900-f2a5-4f65-881e-1374187dd4f9"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC A continuación,vamos a especificar el uso de Hyperopt dentro del flujo.
# MAGIC
# MAGIC En primer lugar, definimos una **función objetivo**. Esta debe tener 2 componentes:
# MAGIC
# MAGIC 1. Un **input** **`params`** que incluye los hiperparametros del espacio de busqueda.
# MAGIC 2. Un **output** que define la metrica que vamos a evaluar.
# MAGIC
# MAGIC En este caso, vamos a usar  **`max_depth`** y **`num_trees`** como hiperparametros y el RMSE como metrica a minimizar. 

# COMMAND ----------

def objective_function(params):    
    
    max_depth = params["max_depth"]
    num_trees = params["num_trees"]

    with mlflow.start_run():
        estimator = pipeline.copy({rf.maxDepth: max_depth, rf.numTrees: num_trees})
        model = estimator.fit(train_df)

        preds = model.transform(val_df)
        rmse = regression_evaluator.evaluate(preds)
        mlflow.log_metric("rmse", rmse)

    return rmse

# COMMAND ----------

# MAGIC %md <i18n value="d4f9dd2b-060b-4eef-8164-442b2be242f4"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Definimos el espacio de busqueda:
# MAGIC
# MAGIC Esto es semejante a los que hacemos con las grillas en grid o randomsearch con la diferencia de que no definimos valores individuales sino el rango. Hyperopt se encarga de encontrar los valores concretos. 

# COMMAND ----------

from hyperopt import hp

search_space = {
    "max_depth": hp.quniform("max_depth", 2, 5, 1),
    "num_trees": hp.quniform("num_trees", 10, 100, 1)
}

# COMMAND ----------

# MAGIC %md <i18n value="27891521-e481-4734-b21c-b2c5fe1f01fe"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC **`fmin()`** genera nuevas configuraciones de hiperparametros para la **función objetivo**. En este caso, evaluara 4 modelos en total usando la información previa disponible en cada ejecución al momento de escoger los nuevos hiperparametros de busqueda. 
# MAGIC
# MAGIC Hyperopt  permite la busqueda en paralelo mediante los metodos de random search o TPE (Tree Parzen Estimators). En este caso usamos TPE ya que en la <a href="http://hyperopt.github.io/hyperopt/scaleout/spark/" target="_blank">documentación</a>, TPE es un metodo adapativo de inspiración bayesiana que hace un mejor uso del espacio de busqueda al usar la información obtenida previamente. 
# MAGIC
# MAGIC Vemos tambien como Hyperopt se integra facilmente con MlFlow. 

# COMMAND ----------

from hyperopt import fmin, tpe, Trials
import numpy as np
import mlflow
import mlflow.spark
mlflow.pyspark.ml.autolog(log_models=False)

num_evals = 4
trials = Trials()
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))

with mlflow.start_run():
    best_max_depth = best_hyperparam["max_depth"]
    best_num_trees = best_hyperparam["num_trees"]
    estimator = pipeline.copy({rf.maxDepth: best_max_depth, rf.numTrees: best_num_trees})
    combined_df = train_df.union(val_df) 

    pipeline_model = estimator.fit(combined_df)
    pred_df = pipeline_model.transform(test_df)
    rmse = regression_evaluator.evaluate(pred_df)

    # Log param and metrics for the final model
    mlflow.log_param("maxDepth", best_max_depth)
    mlflow.log_param("numTrees", best_num_trees)
    mlflow.log_metric("rmse", rmse)
    mlflow.spark.log_model(pipeline_model, "model")
