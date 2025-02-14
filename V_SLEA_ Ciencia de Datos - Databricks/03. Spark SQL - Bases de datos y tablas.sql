-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Tablas y bases de datos
-- MAGIC - Una base de datos es una colección lógica de tablas.  
-- MAGIC - Una tabla pertenece a una base de datos (o esquema, en Databricks son análogos), es una **colección de datos estructurados**. Sus datos persisten **físicamente**.  
-- MAGIC - La metadata de estas tablas y bases, es decir los datos sobre las mismas: nombre, columnas, tipos de datos, etcétera, se almacenan en un [**Metastore**](https://docs.databricks.com/data/metastores/index.html#) gestionado por Databricks.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Las bases de datos se crean con el comando **CREATE SCHEMA**.  
-- MAGIC Podemos agregar *IF NOT EXISTS* para que no falle si la base ya existe.

-- COMMAND ----------

CREATE SCHEMA IF NOT EXISTS datalytics_databricks

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Creación de tablas
-- MAGIC #### [Sintaxis](https://docs.databricks.com/spark/latest/spark-sql/language-manual/sql-ref-syntax-ddl-create-table-datasource.html)

-- COMMAND ----------

CREATE TABLE [ IF NOT EXISTS ] esquema_tabla.nombre_tabla
  [ ( nombre_columna tipo_dato) ]
  USING [TEXT, AVRO, CSV, JSON, JDBC, PARQUET, ORC, HIVE, DELTA, or LIBSVM] -- formatos nativos
  [ OPTIONS ( key1 [ = ] val1, key2 [ = ] val2, ... ) ] -- estas opciones dependen del formato elegido
  [ LOCATION ruta ] -- si se especifica esta opción, la tabla es EXTERNA

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Ejemplo
-- MAGIC Para graficar la creación de tablas, y aún más, para tener datos disponibles para trabajar utilizaremos un dataset público provisto por Databricks. En él encontramos datos de facturación de una empresa de ventas online, con información de: el producto adquirido, el cliente que efectuó la compra y, por supuesto, la fecha.  
-- MAGIC Podemos imaginar que a nuestra empresa ficticia le es de sumo interés analizar las ventas, en especial desean entender si los productos se ven afectados por la estacionalidad, es decir si las ventas varían, para bien o mal, según determinados momentos del año.

-- COMMAND ----------

-- Mostrar que pasa con y sin las options

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Es importante destacar que para nuestro ejemplo, la tabla apunta a un archivo individual. En la práctica, por lo general, las tablas apuntan a una carpeta con varios archivos de igual estructura. De esta manera, definimos el esquema de los datos una única vez y podemos aprovechar el almacenamiento a escala y procesamiento distribuido.  
-- MAGIC Como el archivo está en formato CSV, utilizamos la siguiente sentencia para crear una tabla que nos permita acceder mediante SQL a los datos que contiene:  

-- COMMAND ----------

drop table datalytics_databricks.online_retail;
CREATE TABLE datalytics_databricks.online_retail
USING csv
OPTIONS (header="true",inferschema="true")
LOCATION '/databricks-datasets/definitive-guide/data/retail-data/all/online-retail-dataset.csv'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Con el comando *DESCRIBE TABLE* vemos todas las columnas de la tabla con nombre y tipo de datos.  
-- MAGIC Si no inferimos esquema, los tipos de datos serán *string*, de lo contrario, Spark inferirá el tipo según los datos que encuentre.

-- COMMAND ----------

DESCRIBE TABLE datalytics_databricks.online_retail;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Chequeamos que la tabla se haya creado correctamente.

-- COMMAND ----------

SELECT * FROM datalytics_databricks.online_retail LIMIT 10

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### [Tablas internas (managed) vs externas (unmanaged)](https://docs.databricks.com/data/tables.html#managed-and-unmanaged-tables&language-sql)
-- MAGIC
-- MAGIC Cuando se crea una tabla en Spark, la misma puede ser interna o externa. 
-- MAGIC En resumen, las diferencias entre ambas son:
-- MAGIC
-- MAGIC | Funcionalidad / Tipo Tabla | Interna     | Externa    |
-- MAGIC | ----------------------     | ----------- | ---------- |
-- MAGIC | Gestión Metadatos          | Databricks  | Databricks |
-- MAGIC | Gestión Datos              | Usuario     | Databricks |
-- MAGIC | Borrado Datos con DROP     | No          | Si         |
-- MAGIC
-- MAGIC Generalmente, como los datos con los que trabajaremos residirán en un Data Lake, más específicamente en un servicio de almacenamiento de objetos como AWS S3 o Azure ADLS, usaremos tablas **externas**.
-- MAGIC Sólo se recomienda el uso de tablas internas para pruebas y datos temporales.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Usamos de nuevo el comando *DESCRIBE TABLE*, esta vez incorporamos *EXTENDED* para ver el tipo de tabla que creamos antes.

-- COMMAND ----------

DESCRIBE TABLE EXTENDED datalytics_databricks.online_retail;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Ahora creamos una tabla interna, a partir de la externa ya creada. Luego chequeamos su tipo y la ubicación de los datos.

-- COMMAND ----------

drop table datalytics_databricks.online_retail_interna

-- COMMAND ----------

CREATE TABLE datalytics_databricks.online_retail_interna
USING CSV
AS 
SELECT *
FROM datalytics_databricks.online_retail

-- COMMAND ----------

DESCRIBE TABLE EXTENDED datalytics_databricks.online_retail_interna;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Chequeamos los archivos creados en la ruta de la tabla interna. El directorio /user/hive/warehouse se crea por defecto en **DBFS** para almacenar datos de tablas internas.

-- COMMAND ----------

-- MAGIC %fs ls "/user/hive/warehouse/datalytics_databricks.db/online_retail_interna/"

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Si borramos la tabla y volvemos a chequear la ruta, veremos que ya no existe.

-- COMMAND ----------

DROP TABLE datalytics_databricks.online_retail_interna;

-- COMMAND ----------

-- MAGIC %fs ls "/user/hive/warehouse/datalytics_databricks.db/online_retail_interna/"

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Lectura de datos
-- MAGIC Al hacer un SELECT sobre la tabla, Spark accede al archivo csv y nos devuelve los datos correspondientes.  
-- MAGIC Con un WHERE podemos filtrar y **leer sólo los datos que necesitamos**, reduciendo la cantidad de filas y por ende, **mejorando el tiempo de respuesta**.  
-- MAGIC También mejoraremos este tiempo de respuesta, si, en el SELECT, detallamos sólo las columnas a utilizar. En otras palabras, **evitar el uso del SELECT * **.

-- COMMAND ----------

SELECT customerid,country 
/* 
- SQL NO es case sensitive
- Son buenas prácticas: seleccionar sólo las columnas necesarias y, comentar nuestro código
*/
FROM datalytics_databricks.online_retail
WHERE stockcode='22837' -- seleccionamos un producto de interés

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Escritura de datos
-- MAGIC La sentencia para escribir datos con SQL es **INSERT**. 
-- MAGIC Tenemos diversas [opciones](https://docs.databricks.com/spark/latest/spark-sql/language-manual/sql-ref-syntax-dml-insert.html) disponibles según lo que se necesite resolver.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Inserción incremental
-- MAGIC En este caso sólo añadimos nuevos datos a la tabla, sin alterar los datos ya existentes.

-- COMMAND ----------

-- Sintaxis
INSERT INTO [ TABLE ] esquema_tabla.nombre_tabla [ ( lista_columnas ) ]
    { VALUES ( { value | NULL } [ , ... ] ) [ , ( ... ) ] | query } -- Podemos insertar valores directamente o desde una sub consulta.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Vamos a insertar nuevos datos en nuestra tabla *interna*.  
-- MAGIC **Nota**: no podemos insertar en la *externa* porque, al ser un dataset público, no vamos a tener permisos para escribir en el servicio de almacenamiento correspondiente.

-- COMMAND ----------

INSERT INTO datalytics_databricks.online_retail_interna 
(InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country) -- no es obligatorio especificar las columnas, pero si una buena práctica
VALUES (999999,'ABC123','Un nuevo producto',10,current_timestamp,9.99,17850,'Argentina');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Chequeamos que se haya insertado el registro:

-- COMMAND ----------

SELECT *
FROM datalytics_databricks.online_retail_interna 
WHERE InvoiceNo=999999

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Inserción con sobrescritura
-- MAGIC Primero se borran los datos ya existentes y luego se insertan los nuevos.  
-- MAGIC La sintaxis es igual a la anterior pero se agrega la palabra clave **OVERWRITE** y se quita *INTO*.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Hacemos el mismo INSERT pero con OVERWRITE, vemos que queda sólo este nuevo registro.

-- COMMAND ----------

INSERT OVERWRITE datalytics_databricks.online_retail_interna 
(InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country) -- no es obligatorio especificar las columnas, pero si una buena práctica
VALUES (999999,'ABC123','Un nuevo producto',10,current_timestamp,9.99,17850,'Argentina')

-- COMMAND ----------

SELECT *
FROM datalytics_databricks.online_retail_interna

-- COMMAND ----------

-- MAGIC %md
-- MAGIC También podemos insertar desde una consulta.
-- MAGIC

-- COMMAND ----------

INSERT OVERWRITE datalytics_databricks.online_retail_interna
SELECT * FROM datalytics_databricks.online_retail

-- COMMAND ----------

SELECT COUNT(*) 
FROM datalytics_databricks.online_retail_interna
