# Databricks notebook source
# MAGIC %md
# MAGIC ## Databricks ML
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Identify when a standard cluster is preferred over a single-node cluster and viceversa
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC https://community.databricks.com/t5/data-engineering/when-should-i-use-single-node-clusters-vs-standard/td-p/24301
# MAGIC
# MAGIC Single-node, like the name implies, is a single machine. It still has Spark, just a local cluster. This is a good choice if you are running a workload that does not use Spark, or only needs it for data access. One good example is a small deep learning job. Often it's most efficient and easiest to use one machine with multiple GPUs rather than try to distribute on a cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC Single node clusters are not recommended for large-scale parallel data processing. If you exceed the resources on a single node cluster, a multi node Dataproc cluster is recommended. Single node clusters are not available with high-availability since there is only one node in the cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ###### **Clúster Estándar**
# MAGIC - **Uso Recomendado:**
# MAGIC   - **Procesamiento Distribuido:** Cuando necesitas ejecutar tareas que requieren procesamiento distribuido, como el manejo de grandes volúmenes de datos o la ejecución de algoritmos de aprendizaje automático en paralelo.
# MAGIC   - **Escalabilidad:** Cuando esperas un alto nivel de concurrencia y necesitas que el clúster escale automáticamente para manejar la carga.
# MAGIC   - **Colaboración:** Cuando múltiples usuarios necesitan acceder y trabajar simultáneamente en el clúster.
# MAGIC   - **Tareas en Producción:** Es ideal para cargas de trabajo de producción donde la alta disponibilidad y la tolerancia a fallos son cruciales.
# MAGIC
# MAGIC ###### **Clúster de Nodo Único**
# MAGIC - **Uso Recomendado:**
# MAGIC   - **Desarrollo y Pruebas:** Ideal para desarrollo y pruebas de código, experimentación rápida o análisis de datos en un entorno controlado sin necesidad de recursos distribuidos.
# MAGIC   - **Costo Eficiente:** Es más económico que un clúster estándar porque no utiliza múltiples nodos, lo que lo hace adecuado para tareas de bajo costo y no intensivas en recursos.
# MAGIC   - **Trabajo en Solitario:** Cuando solo un usuario necesita ejecutar tareas sin requerir procesamiento distribuido.
# MAGIC
# MAGIC ###### **Resumen:**
# MAGIC - **Clúster Estándar:** Es mejor cuando necesitas escalabilidad, procesamiento distribuido y colaboración entre múltiples usuarios.
# MAGIC - **Clúster de Nodo Único:** Es preferible para tareas de desarrollo, pruebas o trabajos que no requieren procesamiento distribuido, lo que ayuda a reducir costos.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Connect a repo from an external Git provider to Databricks repos.

# COMMAND ----------

# MAGIC %md
# MAGIC github 
# MAGIC
# MAGIC necesitamos dos cosas: 1. nombre de ususario desde el github y un token de acceso personal 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC https://www.youtube.com/watch?v=qI92bZnz89U
# MAGIC
# MAGIC https://docs.databricks.com/en/repos/git-operations-with-repos.html#clone-a-repo-connected-to-a-remote-git-repository

# COMMAND ----------

# MAGIC %md
# MAGIC #### Commit changes from a Databricks Repo to an external Git provider.

# COMMAND ----------

# MAGIC %md
# MAGIC https://docs.databricks.com/en/repos/git-operations-with-repos.html#commit-and-push-changes-to-the-remote-git-repository

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create a new branch and commit changes to an external Git provider.

# COMMAND ----------

# MAGIC %md
# MAGIC https://docs.databricks.com/en/repos/git-operations-with-repos.html#create-a-new-branch

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pull changes from an external Git provider back to a Databricks workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC Error pulling branch
# MAGIC The Azure account is disabled
# MAGIC Need help? See our Repos Limitations and FAQs docs
# MAGIC
# MAGIC
# MAGIC https://docs.databricks.com/en/repos/git-operations-with-repos.html#pull-changes-from-the-remote-git-repository

# COMMAND ----------

# MAGIC %md
# MAGIC #### Orchestrate multi-task ML workflows using Databricks jobs.

# COMMAND ----------

# MAGIC %md
# MAGIC pendiente

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks Runtime for Machine Learning

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create a cluster with the Databricks Runtime for Machine Learning.

# COMMAND ----------

# MAGIC %md
# MAGIC https://docs.databricks.com/en/machine-learning/index.html#databricks-runtime-for-machine-learning

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Install a Python library to be available to all notebooks that run on a cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/libraries/package-repositories#--pypi-package
# MAGIC
# MAGIC Para instalar una biblioteca de Python que esté disponible para todos los notebooks que se ejecuten en un clúster en Databricks, puedes seguir estos pasos:
# MAGIC
# MAGIC 1. **Accede a la configuración del clúster**:
# MAGIC    - Ve a la interfaz de Databricks.
# MAGIC    - En la barra lateral, selecciona **Compute** para ver los clústeres.
# MAGIC    - Haz clic en el nombre del clúster donde deseas instalar la biblioteca.
# MAGIC
# MAGIC 2. **Instalar la biblioteca**:
# MAGIC    - En la pestaña de configuración del clúster, selecciona **Libraries** (Bibliotecas).
# MAGIC    - Haz clic en **Install New** (Instalar nueva).
# MAGIC    - En el cuadro de diálogo, selecciona **PyPI** como el origen de la biblioteca.
# MAGIC    - Escribe el nombre de la biblioteca que deseas instalar (por ejemplo, `numpy` o `pandas`) y haz clic en **Install**.
# MAGIC
# MAGIC 3. **Reiniciar el clúster (opcional)**:
# MAGIC    - Después de instalar la biblioteca, es posible que necesites reiniciar el clúster para que la instalación surta efecto en todos los notebooks.
# MAGIC
# MAGIC Este proceso hará que la biblioteca esté disponible para todos los notebooks que se ejecuten en ese clúster.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoML

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Identify the steps of the machine learning workflow completed by AutoML.

# COMMAND ----------

# MAGIC %md
# MAGIC Investigar

# COMMAND ----------

# MAGIC %md
# MAGIC There are generally eight steps in the AutoML process: data ingestion, data preparation, data engineering, model selection, model training, hyperparameter tuning, model deployment, and model updates.  https://www.akkio.com/post/how-does-automated-machine-learning-work#:~:text=There%20are%20generally%20eight%20steps,model%20deployment%2C%20and%20model%20updates.

# COMMAND ----------

# MAGIC %md
# MAGIC https://learn.microsoft.com/es-es/training/paths/build-operate-machine-learning-solutions-azure-databricks/ 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC AutoML (Automated Machine Learning) automates several steps of the machine learning workflow, which can significantly speed up the process of developing models. Here are the steps typically completed by AutoML:
# MAGIC
# MAGIC 1. **Data Preprocessing**:
# MAGIC    - **Data Cleaning**: Handling missing values, correcting data types, and dealing with inconsistencies.
# MAGIC    - **Feature Engineering**: Creating new features, encoding categorical variables, normalizing/scaling numerical features.
# MAGIC    - **Data Splitting**: Automatically splitting the data into training, validation, and test sets.
# MAGIC
# MAGIC 2. **Model Selection**:
# MAGIC    - **Algorithm Selection**: Automatically selecting the best algorithms to use based on the problem type (classification, regression, etc.).
# MAGIC    - **Hyperparameter Tuning**: Searching for the optimal hyperparameters for the chosen algorithms.
# MAGIC
# MAGIC 3. **Model Training**:
# MAGIC    - **Model Training**: Training multiple models with different algorithms and hyperparameters.
# MAGIC    - **Cross-Validation**: Automatically performing cross-validation to assess model performance.
# MAGIC
# MAGIC 4. **Model Evaluation**:
# MAGIC    - **Model Scoring**: Evaluating models using appropriate metrics (accuracy, precision, recall, etc.).
# MAGIC    - **Model Ranking**: Ranking models based on their performance on the validation set.
# MAGIC
# MAGIC 5. **Model Selection**:
# MAGIC    - **Best Model Selection**: Selecting the best model based on evaluation metrics.
# MAGIC
# MAGIC 6. **Model Deployment** (optional):
# MAGIC    - **Deployment**: Some AutoML tools also provide options for deploying the best model directly into production environments.
# MAGIC
# MAGIC AutoML handles these tasks to varying degrees, depending on the specific platform or tool you're using, which can greatly simplify the process of developing and deploying machine learning models.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Identify how to locate the source code for the best model produced by AutoML

# COMMAND ----------

# MAGIC %md
# MAGIC El output del código me dirige al notebook, hay otra forma ???

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import random

# Crear una sesión de Spark
spark = SparkSession.builder.appName("AutoML Example").getOrCreate()

# Generar datos ficticios
data = [(random.randint(50, 500),  # price
         random.randint(1, 5),     # bedrooms
         random.randint(1, 3),     # bathrooms
         random.randint(300, 3500), # square_feet
         random.choice(['San Francisco', 'Los Angeles', 'New York', 'Boston'])) # location
        for _ in range(1000)]

# Definir columnas
columns = ["price", "bedrooms", "bathrooms", "square_feet", "location"]

# Crear el DataFrame de ejemplo
airbnb_df = spark.createDataFrame(data, columns)

# Dividir en datos de entrenamiento y prueba
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# Ejecutar AutoML en Databricks
from databricks import automl

summary = automl.regress(train_df, target_col="price", primary_metric="rmse", timeout_minutes=5, max_trials=10)


# COMMAND ----------

# MAGIC %md
# MAGIC #### ● Identify which evaluation metrics AutoML can use for regression problems.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Cuando intento lanzar un ml me da las opciones de: MAE, MSE, RMSE, R-squared

# COMMAND ----------

# MAGIC %md
# MAGIC https://learn.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml?view=azureml-api-2#regressionforecasting-metrics   Article
# MAGIC 08/28/2024
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Identify the key attributes of the data set using the AutoML data exploration notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ???

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Store

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Describe the benefits of using Feature Store to store and access features for machine learning pipelines.

# COMMAND ----------

# MAGIC %md
# MAGIC
