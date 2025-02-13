# Databricks notebook source
# MAGIC %md <i18n value="60a5d18a-6438-4ee3-9097-5145dc31d938"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Introducción Spark ML
# MAGIC Exploraremos el uso de la API de SparkMl para la construcción de modelos
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) En esta lección:<br>
# MAGIC  - Transformación de variables categoricas
# MAGIC  - Uso de la API de SparkMl para Python
# MAGIC  - Almacenamiento y consumo de modelos

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ¿Por que Spark para ML?
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://lh3.googleusercontent.com/d/1iz6DhY4uDYF9MCDW_kCwqOJfJvs3DoQM" width="800"/>
# MAGIC </div>
# MAGIC
# MAGIC ¿Cómo usamos Spark para Ml?
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://lh3.googleusercontent.com/d/1uLV8MPM0s25-2-4aW_4AbgHTY7K-RA81" width="800"/>
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

paths_base = "mnt/testdatabricks/datasets_andercol_test/"
working_dir = "dbfs:/mnt/testdatabricks/datasets_andercol_test/airbnb"

# COMMAND ----------

file_path = f"/{paths_base}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)

# COMMAND ----------

display(airbnb_df)

# COMMAND ----------

# MAGIC %md <i18n value="f8b3c675-f8ce-4339-865e-9c64f05291a6"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Train/Test Split
# MAGIC
# MAGIC El primer paso en la construcción de muchos modelos es la separación en conjuntos de prueba y validación:
# MAGIC

# COMMAND ----------

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol="bedrooms", labelCol="price")
lr_model = lr.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Los componentes basicos del ciclo de vida del modelamiento mendiante Spark son los siguientes (estan inspirados fuertemente en la API de Sklearn):
# MAGIC
# MAGIC - DataFrame: La API de Spark que usaremos utiliza la estructura Dataframe de Spark SQL como fuente principal para entrenar y predecir.
# MAGIC - Transformer: Un Transformer es un algoritmo que puede transformar un dataframe en otro dataframe. Por ejemplo,un modelo es un Transformer que convierte un DataFrame con covariables en un DataFrame con predicciones.
# MAGIC - Estimator: Un Estimator es un algoritmo que puede aprender (.fit()) sobre un DataFrame para producir un Transformer. 
# MAGIC - Pipeline: Un Pipeline encadena multiples Estimadores y Transformadores para producir un flujo de trabajo de modelamiento completo.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC A continuación miremos el uso encadenado de Estimadores y Transformadores. 
# MAGIC

# COMMAND ----------

# MAGIC %md <i18n value="09003d63-70c1-4fb7-a4b7-306101a88ae3"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Variables categoricas
# MAGIC
# MAGIC Algunas ideas para transformar variables categoricas pueden ser:
# MAGIC * Label encoding
# MAGIC * La generación de variables dummies (One Hot Encoding)
# MAGIC * El uso de embeddings vectoriales muntidimiensionales (principalmente para datos de texto o categorias muy dispersas)
# MAGIC
# MAGIC ### One Hot Encoder
# MAGIC En nuestro caso, para explorar la API de SparkML usaremos OHE. Spark no tiene una función que cree las variables dummies directamente por que tenemos que proceder en dos pasos (encadenamos dos Transformers): 
# MAGIC 1. Usamos el <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html?highlight=stringindexer#pyspark.ml.feature.StringIndexer" target="_blank">StringIndexer</a> para mapear una columna de tipo string a labels numericos.  
# MAGIC 2. Aplicamos <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html?highlight=onehotencoder#pyspark.ml.feature.OneHotEncoder" target="_blank">OneHotEncoder</a> al output del StringIndexer

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
ohe_output_cols = [x + "OHE" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
ohe_encoder = OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols)

# COMMAND ----------

display(string_indexer.fit(train_df).transform(test_df))

# COMMAND ----------

display(ohe_encoder.fit(train_df).transform(test_df))

# COMMAND ----------

# MAGIC %md <i18n value="dedd7980-1c27-4f35-9d94-b0f1a1f92839"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Vector Assembler
# MAGIC
# MAGIC Podemos combinar las variables categoricas transformadas con las numericas.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = ohe_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

df_t1 = string_indexer.fit(train_df).transform(train_df)
df_t2 = ohe_encoder.fit(df_t1).transform(df_t1)
df_t3 = vec_assembler.transform(df_t2)

# COMMAND ----------

display(df_t1)

# COMMAND ----------

display(df_t2)

# COMMAND ----------

display(df_t3)

# COMMAND ----------

# MAGIC %md
# MAGIC Spark usa por defecto tipos de datos como vectores dispersos en lugar de densos. 

# COMMAND ----------

# MAGIC %md <i18n value="fb06fb9b-5dac-46df-aff3-ddee6dc88125"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Linear Regression
# MAGIC
# MAGIC Finalmente, podemos definir el modelo que vamos a estimar

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor

lr = LinearRegression(labelCol="price", featuresCol="features")
dt = DecisionTreeRegressor(labelCol="price", featuresCol="features")

# COMMAND ----------

# MAGIC %md <i18n value="a7aabdd1-b384-45fc-bff2-f385cc7fe4ac"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Pipeline
# MAGIC
# MAGIC Podemos ensamblar todos los pasos en un <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html?highlight=pipeline#pyspark.ml.Pipeline" target="_blank">Pipeline</a>
# MAGIC
# MAGIC De esta manera, creamos procesos reproducibles. 

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [string_indexer, ohe_encoder, vec_assembler, lr]
pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="c7420125-24be-464f-b609-1bb4e765d4ff"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Almancenamiento de modelos
# MAGIC
# MAGIC Podemos guardar los modelos en un almacenamiento persistente como un lago de datos. De forma que lo podemos reutilizar en procesos posteriores. 

# COMMAND ----------

working_dir

# COMMAND ----------

pipeline_model.write().overwrite().save(working_dir + "/modelo")

# COMMAND ----------

# MAGIC %md <i18n value="15f4623d-d99a-42d6-bee8-d7c4f79fdecb"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Consumo de modelos
# MAGIC
# MAGIC Los modelos tienen metadata vinculada al tipo de objeto usado. Por esta razon, debemos recordar el tipo de modelo al usarlo
# MAGIC
# MAGIC Por eso, se recomienda el uso de pipelines ya que el objeto Pipeline es capaz de obtener el flujo completo. 

# COMMAND ----------

from pyspark.ml import PipelineModel

saved_pipeline_model = PipelineModel.load(working_dir + '/modelo')

# COMMAND ----------

# MAGIC %md <i18n value="1303ef7d-1a57-4573-8afe-561f7730eb33"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Inferencia sobre datos de prueba

# COMMAND ----------

pred_df = saved_pipeline_model.transform(test_df)

display(pred_df.select("features", "price", "prediction"))

# COMMAND ----------

# MAGIC %md <i18n value="9497f680-1c61-4bf1-8ab4-e36af502268d"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluación de modelos
# MAGIC
# MAGIC Los metodos de evaluación tambien son un Transformer

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
