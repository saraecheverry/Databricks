# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

# Check if the file exists
dbutils.fs.ls("dbfs:/user/hive/warehouse/")

# COMMAND ----------

# Ruta al archivo en DBFS
delta_path = "dbfs:/user/hive/warehouse/ft_clientes_productos_csv"

# Leer los datos usando el formato Delta
df_spark = spark.read.format("delta").load(delta_path)

# Mostrar los primeros registros
display(df_spark)
ft_clientes_productos = df_spark.toPandas()


# COMMAND ----------

# Ruta al archivo en DBFS
delta_path = "dbfs:/user/hive/warehouse/dim_unidad_producto_csv"

# Leer los datos usando el formato Delta
df_spark = spark.read.format("delta").load(delta_path)
dim_unidad_producto = df_spark.toPandas()


# COMMAND ----------

dim_unidad_producto.shape

# COMMAND ----------

# ft_clientes_productos_ = ft_clientes_productos[ft_clientes_productos['fk_cliente'].isin([339132, 282953])]
# ft_clientes_productos_ = ft_clientes_productos.sample(frac=0.5, random_state=42)
ft_clientes_productos_ = ft_clientes_productos.copy()

# COMMAND ----------

import pandas as pd

# Agrupar por 'fk_cliente' y 'fk_fecha_servicio' y unir los 'fk_producto' separados por comas
ft_clientes_productos_grp = ft_clientes_productos_.groupby(['fk_cliente', 'fk_fecha_servicio'], as_index=False).agg({
    'fk_producto': lambda x: ','.join(sorted(set(x.astype(str)))),  # Unir valores únicos
    # 'pk_unidad_marca': lambda x: ','.join(sorted(set(x.astype(str)))),  # Unir valores únicos
    'cantidad_producto': 'sum',  # Sumar la cantidad de productos
    # Agregar más columnas si es necesario
})
ft_clientes_productos_grp.rename(columns={'cantidad_producto':'cantidad_producto_dia'}, inplace=True)
ft_clientes_productos_grp

# COMMAND ----------

# MAGIC %md
# MAGIC ### con paquetes por 2 meses

# COMMAND ----------

# Función para eliminar duplicados dentro de cada elemento
def eliminar_duplicados(lista):
    nueva_lista = []
    for item in lista:
        # Separar los elementos por comas si las tiene
        elementos = item.split(',')
        # Usar un conjunto para eliminar duplicados, luego volver a unir los elementos
        elementos_unicos = ','.join(sorted(set(elementos)))
        # Agregar el resultado a la nueva lista
        nueva_lista.append(elementos_unicos)
    return nueva_lista

# Función para verificar la diferencia de fechas con el registro anterior
def check_date_diff(group):
    group = group.sort_values('fk_fecha_servicio')
    # Crear una columna de diferencia de días con respecto al registro anterior
    group['date_diff'] = group['fk_fecha_servicio'].diff().dt.days
    group['existe_registro_menos_60_dias'] = group['date_diff'] < 60
    group['existe_registro_menos_60_dias'].fillna(False, inplace=True)
    return group

# COMMAND ----------

import pandas as pd

# Asegúrate de que la columna 'fk_fecha_servicio' esté en formato de fecha
ft_clientes_productos_grp['fk_fecha_servicio'] = pd.to_datetime(ft_clientes_productos_grp['fk_fecha_servicio'])

# Ordenar el DataFrame por cliente y fecha
df = ft_clientes_productos_grp.sort_values(by=['fk_cliente', 'fk_fecha_servicio'])
# df = df[df['fk_cliente'] == 393291]
# Crear las columnas 'paquete_items' y 'cantidad_producto'
paquete_items_list = []
cantidad_producto_list = []

# Iterar sobre cada cliente para crear los paquetes de productos
for cliente, group in df.groupby('fk_cliente'):
    # print(group)
    group = group.sort_values(by='fk_fecha_servicio')  # Asegurar que cada cliente esté ordenado por fecha
    paquete_items_cliente = []  # Lista para guardar los paquetes de cada cliente
    cantidad_producto_cliente = []  # Lista para guardar la cantidad de productos
    
    for i in range(len(group)):
        fecha_inicial = group['fk_fecha_servicio'].iloc[i]  # Fecha de servicio actual
        # Filtrar productos dentro de los próximos 60 días
        productos_en_rango = group[(group['fk_fecha_servicio'] >= fecha_inicial) &
                                   (group['fk_fecha_servicio'] <= fecha_inicial + pd.Timedelta(days=60))]
        # print('productos_en_rango',productos_en_rango)
        # Obtener los fk_productos como una lista
        productos = productos_en_rango['fk_producto'].astype(str).tolist()
        # Concatenar los productos en una cadena separados por coma
        productos = ','.join(productos)
        paquete_items_cliente.append(productos)
        # Contar la cantidad de productos encontrados
        cantidad_producto_cliente.append(len(productos.split(',') ))
    
    # Agregar los paquetes y cantidades generadas a las listas finales
    paquete_items_list.extend(paquete_items_cliente)
    cantidad_producto_list.extend(cantidad_producto_cliente)

# Añadir las nuevas columnas 'paquete_items' y actualizar 'cantidad_producto'
df['paquete_items'] = eliminar_duplicados(paquete_items_list)
df['cantidad_producto_paq'] = cantidad_producto_list

# COMMAND ----------

# Aplicar la función en cada grupo de fk_cliente
df = df.groupby('fk_cliente').apply(check_date_diff).reset_index(drop=True)

# COMMAND ----------

df = df[df['existe_registro_menos_60_dias'] == False]
df['id_transaccion'] = range(1, len(df) + 1)
df

# COMMAND ----------

# Crear una matriz de unos y ceros con pd.get_dummies() y agrupar por transacción
df_matrix = df.groupby('id_transaccion')['paquete_items'].apply(lambda x: ','.join(x.astype(str)))
df_matrix = df_matrix.str.get_dummies(sep=',')
df_matrix

# COMMAND ----------

df_matrix = df_matrix.astype(bool)

# COMMAND ----------



# COMMAND ----------

df_matrix

# COMMAND ----------

for i in df_matrix.columns:
    if '3' in i:
        print(i)

# COMMAND ----------

# pip install mlxtend

# COMMAND ----------

df_matrix.shape

# COMMAND ----------

sum(df_matrix['379'])/ 167773

# COMMAND ----------

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Generar los itemsets frecuentes utilizando Apriori
# Establecemos un soporte mínimo del %
frequent_itemsets = apriori(df_matrix, min_support=0.00000001, use_colnames=True)

# Mostrar los itemsets frecuentes
print("Itemsets frecuentes:")
frequent_itemsets_ = frequent_itemsets.sort_values('support', ascending=False)
frequent_itemsets_['itemsets'] = frequent_itemsets_['itemsets'].apply(lambda x: list(x)[0])
frequent_itemsets_

# COMMAND ----------



# COMMAND ----------

dim_unidad_producto.drop(columns='_c0', inplace=True)

# COMMAND ----------

frequent_itemsets_['itemsets'] = frequent_itemsets_['itemsets'].astype(int)
# frequent_itemsets_.merge(dim_unidad_producto[['fk_producto', 'Producto_final']].drop_duplicates(), left_on = 'itemsets', right_on='fk_producto')
frequent_itemsets_.merge(dim_unidad_producto, left_on = 'itemsets', right_on='pk_producto')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Generar reglas de asociación utilizando confianza mínima del 50%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

# Mostrar las reglas de asociación
print("\nReglas de asociación:")
rules

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Con el paquete de toda la historia

# COMMAND ----------

ft_clientes_productos_grp

# COMMAND ----------

# Agrupar por 'fk_cliente' y 'fk_fecha_servicio' y unir los 'fk_producto' separados por comas
ft_clientes_productos_grp = ft_clientes_productos_.groupby(['fk_cliente'], as_index=False).agg({
    'fk_producto': lambda x: ','.join(sorted(set(x.astype(str)))),  # Unir valores únicos
    # 'pk_unidad_marca': lambda x: ','.join(sorted(set(x.astype(str)))),  # Unir valores únicos
    'cantidad_producto': 'sum',  # Sumar la cantidad de productos
    # Agregar más columnas si es necesario
})
ft_clientes_productos_grp.rename(columns={'cantidad_producto':'cantidad_producto_dia'}, inplace=True)
ft_clientes_productos_grp

# COMMAND ----------

ft_clientes_productos_grp['id_transaccion'] = range(1, len(ft_clientes_productos_grp) + 1)

# COMMAND ----------

# Crear una matriz de unos y ceros con pd.get_dummies() y agrupar por transacción
df_matrix = ft_clientes_productos_grp.groupby('id_transaccion')['fk_producto'].apply(lambda x: ','.join(x.astype(str)))
df_matrix = df_matrix.str.get_dummies(sep=',')
df_matrix

# COMMAND ----------

df_matrix = df_matrix.astype(bool)

# COMMAND ----------

df_matrix

# COMMAND ----------

df_matrix.shape

# COMMAND ----------

sum(df_matrix['14'])/ 82494

# COMMAND ----------

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Generar los itemsets frecuentes utilizando Apriori
# Establecemos un soporte mínimo del %
frequent_itemsets = apriori(df_matrix, min_support=0.001, use_colnames=True)

# Mostrar los itemsets frecuentes
print("Itemsets frecuentes:")
frequent_itemsets_ = frequent_itemsets.sort_values('support', ascending=False)
frequent_itemsets_['itemsets'] = frequent_itemsets_['itemsets'].apply(lambda x: list(x)[0])
frequent_itemsets_

# COMMAND ----------

len(df_matrix.columns)

# COMMAND ----------

frequent_itemsets_[frequent_itemsets_['itemsets'] == 14]

# COMMAND ----------

frequent_itemsets_.dtypes

# COMMAND ----------

df_unique = frequent_itemsets_.loc[frequent_itemsets_.groupby('itemsets')['support'].idxmax()]

# COMMAND ----------

df_unique['itemsets'] = df_unique['itemsets'].astype(int)
# df_unique.merge(dim_unidad_producto[['fk_producto', 'Producto_final']].drop_duplicates(), left_on = 'itemsets', right_on='fk_producto')
df_unique_des = df_unique.merge(dim_unidad_producto, left_on = 'itemsets', right_on='pk_producto').sort_values('support', ascending=False)

# COMMAND ----------

# pip install openpyxl

# COMMAND ----------

import shutil
import os
from pyspark.dbutils import DBUtils

dbutils = DBUtils(spark)

# Write the file to a local temporary path
local_path = '/tmp/df_unique_des_soporte_paqhist.xlsx'
dbfs_path = 'dbfs:/user/hive/warehouse/df_unique_des_soporte_paqhist/df_unique_des_soporte_paqhist.xlsx'
dbfs_dir = 'dbfs:/user/hive/warehouse/df_unique_des_soporte_paqhist/'

# Ensure the target directory exists
dbutils.fs.mkdirs(dbfs_dir)

df_unique_des.to_excel(local_path)

# Move the file from the local path to DBFS
shutil.move(local_path, dbfs_path.replace('dbfs:', '/dbfs'))

# COMMAND ----------



# COMMAND ----------

# Copia el archivo desde DBFS a un lugar accesible públicamente, como /FileStore
dbutils.fs.cp("dbfs:/tmp/df_unique_des_soporte_paqhist.xlsx", "/FileStore/df_unique_des_soporte_paqhist.xlsx")

# Genera un enlace para descargar el archivo
download_link = "/files/df_unique_des_soporte_paqhist.xlsx"
print(f"Download link: {download_link}")

# COMMAND ----------

delta_path = "dbfs:/user/hive/warehouse/ft_clientes_productos_csv"


# COMMAND ----------

delta_path

# COMMAND ----------

# Generar reglas de asociación utilizando confianza mínima del 50%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

# Mostrar las reglas de asociación
print("\nReglas de asociación:")
rules

# COMMAND ----------

dim_unidad_producto

# COMMAND ----------

dim_unidad_producto['Productototal'] = dim_unidad_producto['unidad_negocio'] + '|' +  dim_unidad_producto['marca_negocio'] + '|' +dim_unidad_producto['nombre_producto']+ '|' + dim_unidad_producto['nombre_subproducto'] + '|' +dim_unidad_producto['familia_producto']

# COMMAND ----------

rules_ = rules[['antecedents','consequents','confidence' ]]

rules_['antecedents1'] = rules_['antecedents'].apply(lambda x: list(x)[0])
rules_['antecedents1'] = rules_['antecedents1'].astype(int)
rules_['antecedents2'] = rules_['antecedents'].apply(lambda x: list(x)[1] if len(x) > 1 else None)
rules_['antecedents2'] = pd.to_numeric(rules_['antecedents2'], errors='coerce').astype('Int64')
# Reemplazar NaN con 0
rules_['antecedents2'] = rules_['antecedents2'].fillna(0).astype(int)

rules_['consequents1'] = rules_['consequents'].apply(lambda x: list(x)[0])
rules_['consequents1'] = rules_['consequents1'].astype(int)
rules_['consequents2'] = rules_['consequents'].apply(lambda x: list(x)[1] if len(x) > 1 else None)
rules_['consequents2'] = pd.to_numeric(rules_['consequents2'], errors='coerce').astype('Int64')
# Reemplazar NaN con 0
rules_['consequents2'] = rules_['consequents2'].fillna(0).astype(int)

rules_ = rules_.merge(dim_unidad_producto[['pk_producto','Productototal']], left_on = 'antecedents1', right_on='pk_producto')
rules_.rename(columns={'Productototal':'antecedents1_Productototal'}, inplace=True)
rules_.drop(columns=['pk_producto'], inplace=True)
rules_ = rules_.merge(dim_unidad_producto[['pk_producto','Productototal']], left_on = 'antecedents2', right_on='pk_producto', how='left')
rules_.rename(columns={'Productototal':'antecedents2_Productototal'}, inplace=True)
rules_.drop(columns=['pk_producto'], inplace=True)


rules_ = rules_.merge(dim_unidad_producto[['pk_producto','Productototal']], left_on = 'consequents1', right_on='pk_producto')
rules_.rename(columns={'Productototal':'consequents1_Productototal'}, inplace=True)
rules_.drop(columns=['pk_producto'], inplace=True)
rules_ = rules_.merge(dim_unidad_producto[['pk_producto','Productototal']], left_on = 'consequents2', right_on='pk_producto', how='left')
rules_.rename(columns={'Productototal':'consequents2_Productototal'}, inplace=True)
rules_.drop(columns=['pk_producto'], inplace=True)


rules_

# COMMAND ----------

rules_.to_excel('antecedents_consequents_paqhist.xlsx')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


