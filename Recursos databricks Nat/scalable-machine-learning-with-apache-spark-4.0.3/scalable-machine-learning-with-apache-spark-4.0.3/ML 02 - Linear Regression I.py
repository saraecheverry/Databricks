# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# DBTITLE 1,--i18n-62811f6d-e550-4c60-8903-f38d7ed56ca7
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Regression: Predicting Rental Price
# MAGIC
# MAGIC In this notebook, we will use the dataset we cleansed in the previous lab to predict Airbnb rental prices in San Francisco.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC * Use the SparkML to build a linear regression model
# MAGIC * Identify the differences between estimators and transformers in Spark ML

# COMMAND ----------

# DBTITLE 1,--i18n-1e2c921e-1125-4df3-b914-d74bf7a73ab5
# MAGIC %md
# MAGIC ## 📌 Requirements
# MAGIC
# MAGIC **Required Databricks Runtime Version:** 
# MAGIC * Please note that in order to run this notebook, you must use one of the following Databricks Runtime(s): **14.3.x-scala2.12 14.3.x-photon-scala2.12 14.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# DBTITLE 1,--i18n-6a1bb996-7b50-4f03-9bcd-3d3ec3069a6d
# MAGIC %md
# MAGIC ## Lesson Setup
# MAGIC
# MAGIC The first thing we're going to do is to **run setup script**. This script will define the required configuration variables that are scoped to each user.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 1,--i18n-b44be11f-203c-4ea4-bc3e-20e696cabb0e
# MAGIC %md
# MAGIC ## Load Dataset

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)

# COMMAND ----------

# DBTITLE 1,--i18n-ee10d185-fc70-48b8-8efe-ea2feee28e01
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Train/Test Split
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/301/TrainTestSplit.png)
# MAGIC
# MAGIC **Question**: Why is it necessary to set a seed? What happens if I change my cluster configuration?

# COMMAND ----------

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------

# DBTITLE 1,--i18n-b70f996a-31a2-4b62-a699-dc6026105465
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's change the # of partitions (to simulate a different cluster configuration), and see if we get the same number of data points in our training set.

# COMMAND ----------

train_repartition_df, test_repartition_df = (airbnb_df
                                             .repartition(24)
                                             .randomSplit([.8, .2], seed=42))

print(train_repartition_df.count())

# COMMAND ----------

# DBTITLE 1,--i18n-5b96c695-717e-4269-84c7-8292ceff9d83
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Linear Regression
# MAGIC
# MAGIC We are going to build a very simple model predicting **`price`** just given the number of **`bedrooms`**.
# MAGIC
# MAGIC **Question**: What are some assumptions of the linear regression model?

# COMMAND ----------

display(train_df.select("price", "bedrooms"))

# COMMAND ----------

display(train_df.select("price", "bedrooms").summary())

# COMMAND ----------

display(train_df)

# COMMAND ----------

# DBTITLE 1,--i18n-4171a9ae-e928-41e3-9689-c6fcc2b3d57c
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC There does appear to be some outliers in our dataset for the price ($10,000 a night??). Just keep this in mind when we are building our models.
# MAGIC
# MAGIC We will use <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html?highlight=linearregression#pyspark.ml.regression.LinearRegression" target="_blank">LinearRegression</a> to build our first model.
# MAGIC
# MAGIC The cell below will fail because the Linear Regression estimator expects a vector of values as input. We will fix that with VectorAssembler below.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="bedrooms", labelCol="price")

# Uncomment when running
# lr_model = lr.fit(train_df)

# COMMAND ----------

# DBTITLE 1,--i18n-f1353d2b-d9b8-4c8c-af18-2abb8f0d0b84
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Vector Assembler
# MAGIC
# MAGIC What went wrong? Turns out that the Linear Regression **estimator** (**`.fit()`**) expected a column of Vector type as input.
# MAGIC
# MAGIC We can easily get the values from the **`bedrooms`** column into a single vector using <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler" target="_blank">VectorAssembler</a>. VectorAssembler is an example of a **transformer**. Transformers take in a DataFrame, and return a new DataFrame with one or more columns appended to it. They do not learn from your data, but apply rule based transformations.
# MAGIC
# MAGIC You can see an example of how to use VectorAssembler on the <a href="https://spark.apache.org/docs/latest/ml-features.html#vectorassembler" target="_blank">ML Programming Guide</a>.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")

vec_train_df = vec_assembler.transform(train_df)

# COMMAND ----------

lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(vec_train_df)

# COMMAND ----------

# DBTITLE 1,--i18n-ab8f4965-71db-487d-bbb3-329216580be5
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Inspect the Model

# COMMAND ----------

m = lr_model.coefficients[0]
b = lr_model.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

# COMMAND ----------

# DBTITLE 1,--i18n-ae6dfaf9-9164-4dcc-a699-31184c4a962e
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Apply Model to Test Set

# COMMAND ----------

vec_test_df = vec_assembler.transform(test_df)

pred_df = lr_model.transform(vec_test_df)

pred_df.select("bedrooms", "features", "price", "prediction").show()

# COMMAND ----------

# DBTITLE 1,--i18n-8d73c3ee-34bc-4f8b-b2ba-03597548680c
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluate the Model
# MAGIC
# MAGIC Let's see how our linear regression model with just one variable does. Does it beat our baseline model?

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")

# COMMAND ----------

# DBTITLE 1,--i18n-703fbf0b-a2e1-4086-b002-8f63e06afdd8
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Wahoo! Our RMSE is better than our baseline model. However, it's still not that great. Let's see how we can further decrease it in future notebooks.

# COMMAND ----------

# DBTITLE 1,--i18n-a2c7fb12-fd0b-493f-be4f-793d0a61695b
# MAGIC %md
# MAGIC
# MAGIC ## Classroom Cleanup
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
