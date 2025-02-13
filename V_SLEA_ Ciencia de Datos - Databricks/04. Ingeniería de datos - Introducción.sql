-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Introducción
-- MAGIC En este módulo vamos a utilizar la interfaz interactiva de Databricks para realizar algunas tareas poniéndonos en el rol de un **ingeniero de datos**. Para ello, vamos a continuar trabajando con el dataset visto en el módulo anterior, o lo que es lo mismo, la tabla *online_retail* del esquema *datalytics_databricks*. Nuestro objetivo será generar las siguientes tablas con información sumarizada para que el equipo de visualización construya reportes o tableros con base a estos datos:  
-- MAGIC - Listado de los 3 productos más vendidos por mes.
-- MAGIC - Serie de tiempo para productos de interés. 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Primero volvemos a darle un vistazo a los datos.

-- COMMAND ----------

SELECT * FROM datalytics_databricks.online_retail LIMIT 10

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Agregamos comentarios a las columnas para evitar ambigüedades o malos entendidos.

-- COMMAND ----------

ALTER TABLE datalytics_databricks.online_retail CHANGE COLUMN InvoiceNo   COMMENT "Número de factura";
ALTER TABLE datalytics_databricks.online_retail CHANGE COLUMN StockCode   COMMENT "Código de producto";
ALTER TABLE datalytics_databricks.online_retail CHANGE COLUMN Description COMMENT "Descripción de producto";
ALTER TABLE datalytics_databricks.online_retail CHANGE COLUMN Quantity    COMMENT "Cantidad vendida";
ALTER TABLE datalytics_databricks.online_retail CHANGE COLUMN InvoiceDate COMMENT "Fecha de venta";
ALTER TABLE datalytics_databricks.online_retail CHANGE COLUMN UnitPrice   COMMENT "Precio por unidad";
ALTER TABLE datalytics_databricks.online_retail CHANGE COLUMN CustomerID  COMMENT "Identificador del cliente";
ALTER TABLE datalytics_databricks.online_retail CHANGE COLUMN Country     COMMENT "País del cliente";

-- COMMAND ----------

DESCRIBE TABLE datalytics_databricks.online_retail

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Limpieza de datos
-- MAGIC Cuando nos referimos a *limpiar* los datos, en realidad englobamos una gran cantidad de posibles acciones a realizar, como ser:
-- MAGIC - Filtrado de datos incorrectos.
-- MAGIC - Imputación de datos faltantes.
-- MAGIC - Detección de outliers.
-- MAGIC - Normalización de direcciones.  
-- MAGIC
-- MAGIC Para nuestro caso, vamos a:
-- MAGIC 1. Corroborar si hay datos faltantes y cuantificarlos.
-- MAGIC 2. Imputar los datos faltantes.
-- MAGIC 3. Filtrar datos no relevantes: ventas con unidades negativas, productos con códigos inválidos (menos de 5 dígitos), productos asociados a créditos.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Cuantificación de datos faltantes
-- MAGIC Nos referimos a dato faltante si encontramos un valor **null** en un registro para cualquiera de las columnas de la tabla.

-- COMMAND ----------

SELECT 
ROUND((1-(no_nulos_InvoiceNo/total_registros))*100,2)   as porcentaje_nulos_InvoiceNo,
ROUND((1-(no_nulos_StockCode/total_registros))*100,2)   as porcentaje_nulos_StockCode,
ROUND((1-(no_nulos_Description/total_registros))*100,2) as porcentaje_nulos_Description,
ROUND((1-(no_nulos_Quantity/total_registros))*100,2)    as porcentaje_nulos_Quantity,
ROUND((1-(no_nulos_InvoiceDate/total_registros))*100,2) as porcentaje_nulos_InvoiceDate,
ROUND((1-(no_nulos_UnitPrice/total_registros))*100,2)   as porcentaje_nulos_UnitPrice,
ROUND((1-(no_nulos_CustomerID/total_registros))*100,2)  as porcentaje_nulos_CustomerID,
ROUND((1-(no_nulos_Country/total_registros))*100,2)     as porcentaje_nulos_Country
FROM
(
  SELECT  
    COUNT(InvoiceNo)   AS no_nulos_InvoiceNo,
    COUNT(StockCode)   AS no_nulos_StockCode,
    COUNT(Description) AS no_nulos_Description,
    COUNT(Quantity)    AS no_nulos_Quantity,
    COUNT(InvoiceDate) AS no_nulos_InvoiceDate,
    COUNT(UnitPrice)   AS no_nulos_UnitPrice,
    COUNT(CustomerID)  AS no_nulos_CustomerID,
    COUNT(Country)     AS no_nulos_Country,
    COUNT(*) AS total_registros
  FROM datalytics_databricks.online_retail
) AS cantidad_nulos

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Imputación de datos faltantes
-- MAGIC Como se observa, el campo con mayor cantidad de datos faltantes es **CustomerID**, con un 25%. Debido a que es un número muy alto para descartar esos registros, tomamos la decisión de imputar, reemplazando el **null** por el valor -1, de manera tal que ese ID luego pueda ser joineado con un cliente "inexistente".  
-- MAGIC Por otra parte, hay una pequeña cantidad de productos sin descripción, los descartaremos junto con los registros a filtrar según los criterios especificados anteriormente.

-- COMMAND ----------

-- MAGIC %fs rm -r /user/hive/warehouse/datalytics_databricks.db/online_retail_limpieza

-- COMMAND ----------

-- Creamos una tabla para almacenar los resultados intermedios de la limpieza
CREATE TABLE datalytics_databricks.online_retail_limpieza 
USING PARQUET -- Usamos este formato ya que es más eficiente que un CSV, en el próximo módulo veremos más detalles cuando nos adentremos en el formato DELTA
AS
-- Cambiamos los nombres de las columnas a español y formato underscore
SELECT
  InvoiceNo   AS nro_factura,
  StockCode   AS codigo_producto,
  Description AS descripcion_producto,
  Quantity    AS cantidad_vendida,
  InvoiceDate AS fecha_factura,
  UnitPrice   AS precio_unitario,
  COALESCE(CustomerID,-1) AS id_cliente,
  Country     AS pais_cliente
FROM datalytics_databricks.online_retail
WHERE Description IS NOT NULL
AND Quantity>0
AND LENGTH(StockCode)>5
AND Description NOT LIKE '%credit%'
AND Description <> '?'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Transformación y carga
-- MAGIC A partir de los datos curados, vamos a generar las 2 tablas planteadas al principio.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Top 3 productos más vendidos por mes
-- MAGIC Para obtener este listado, sumarizamos la cantidad de unidades vendidas por descripción de producto y mes. Para obtener este último, primero debemos convertir la fecha_factura que es de tipo *string* a tipo *timestamp*.  
-- MAGIC Luego, hacemos uso de la función ventana [*ROW_NUMBER*](https://docs.databricks.com/sql/language-manual/functions/row_number.html) para calcular el ranking de ventas.

-- COMMAND ----------

-- MAGIC %fs rm -r /user/hive/warehouse/datalytics_databricks.db/top_3_productos

-- COMMAND ----------

/* Usamos la misma sintaxis vista en el módulo anterior para crear la tabla solicitada, sólo que ahora al final le agregamos la palabra clave AS para que use una sentencia SELECT. Databricks usará el resultado devuelto para inferir la estructura e insertar los datos.*/
CREATE TABLE datalytics_databricks.top_3_productos 
USING PARQUET
AS 
WITH ventas_mes AS (
  SELECT
    DATE_FORMAT(
      TO_TIMESTAMP(fecha_factura, 'M/d/yyyy H:mm'),
      'yyyyMM'
    ) AS anio_mes_factura,
    descripcion_producto,
    SUM(cantidad_vendida) AS total_cantidad_vendida
  FROM
    datalytics_databricks.online_retail_limpieza
  GROUP BY
    anio_mes_factura,
    descripcion_producto
)
SELECT
  *
FROM
  (
    SELECT
      anio_mes_factura,
      descripcion_producto,
      total_cantidad_vendida,
      ROW_NUMBER() OVER(
        PARTITION BY anio_mes_factura
        ORDER BY
          total_cantidad_vendida DESC
      ) AS ranking_producto_mes
    FROM
      ventas_mes
  ) ranking
WHERE
  ranking_producto_mes <= 3

-- COMMAND ----------

SELECT * 
FROM datalytics_databricks.top_3_productos 
WHERE anio_mes_factura<=201111
ORDER BY anio_mes_factura

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Serie de tiempo
-- MAGIC A partir del análisis anterior, detectamos que los productos tipo *JUMBO BAG* son de los más vendidos, es por esto que ahora la empresa quiere analizarlos más en detalle para ver si hay alguna época del año donde las ventas se disparen o caigan.

-- COMMAND ----------

CREATE TABLE datalytics_databricks.ventas_historicas_jumbo_bag 
USING PARQUET
AS
SELECT
  DATE_trunc(
    'DD',
    TO_TIMESTAMP(fecha_factura, 'M/d/yyyy H:mm')
  ) AS dia_factura,
  descripcion_producto,
  SUM(cantidad_vendida) AS total_cantidad_vendida
FROM
  datalytics_databricks.online_retail_limpieza
WHERE
  DATE_FORMAT(
    TO_TIMESTAMP(fecha_factura, 'M/d/yyyy H:mm'),
    'yyyyMM'
  ) <= 201111
  and descripcion_producto like 'JUMBO %'
GROUP BY
  dia_factura,
  descripcion_producto
order by
  dia_factura,
  descripcion_producto

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Análisis mensual

-- COMMAND ----------

SELECT
  DATE_trunc('MM', dia_factura) AS anio_mes_factura,
  SUM(total_cantidad_vendida) AS total_cantidad_vendida
FROM
  datalytics_databricks.ventas_historicas_jumbo_bag
GROUP BY
  anio_mes_factura
ORDER BY
  anio_mes_factura

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Se observa un pico de ventas en Marzo, consultando más en detalle vemos que el pico es debido a una gran cantidad de ventas para el día 17. A partir de esta instancia, será tarea de los analistas de determinar el por qué de estas anomalías pero lo importante es que podrán contar con información ya limpia, filtrada y sumarizada. De esta manera pueden concentrar su trabajo en el análisis, sin tener que lidiar con las tareas que ya resolvió el áera de ingeniería de datos.

-- COMMAND ----------

SELECT
  dia_factura,
  descripcion_producto,
  total_cantidad_vendida
FROM
  datalytics_databricks.ventas_historicas_jumbo_bag
  where dia_factura between '2011-03-01' and '2011-03-31'
ORDER BY
  dia_factura
