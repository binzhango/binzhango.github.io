---
title: Spark Structured Streaming

date: 
  created: 2020-02-08 23:23:13

tags:
  - Data Engineer
categories: 
  - Spark
---


## Spark Structured Streaming

Recently reading a blog [Structured Streaming in PySpark](https://hackersandslackers.com/structured-streaming-in-pyspark/)
It's implemented in Databricks platform. Then I try to implement in my local Spark.
Some tricky issue happened during my work.

<!-- more -->

## Reading Data
```python
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType, StringType, StructType, StructField

spark = SparkSession.builder.appName("Test Streaming").enableHiveSupport().getOrCreate()

json_schema = StructType([
    StructField("time", TimestampType(), True),
    StructField("customer", StringType(), True),
    StructField("action", StringType(), True),
    StructField("device", StringType(), True)
])

file_path = "local_file_path<file:///..."

```
#### read json as same as method in the blog
```python
input = spark.read.schema(json_schema).json(file_path)

input.show()
# +----+--------+------+------+
# |time|customer|action|device|
# +----+--------+------+------+
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# +----+--------+------+------+
input.count()
# 20000
```
All values are null, however, the count is right. It means spark has already read all data but the schema is not correctly mapped.

#### read a single json file to check schema
```python
input = spark.read.schema(json_schema).json(file_path+'/1.json')

input.show()

# +----+--------+------+------+
# |time|customer|action|device|
# +----+--------+------+------+
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# |null|    null|  null|  null|
# +----+--------+------+------+

# same error
# Then I drop schema option and use inferSchema
input = spark.read.json(file_path+'/1.json')

input.show()

# +--------------------+-----------+-----------------+--------------------+---------------+
# |     _corrupt_record|     action|         customer|              device|           time|
# +--------------------+-----------+-----------------+--------------------+---------------+
# |[{"time":"3:57:09...|       null|             null|                null|           null|
# |                null|  power off|Nicolle Pargetter| August Doorbell Cam| 1:29:05.000 AM|
# |                null|   power on|   Concordia Muck|Footbot Air Quali...| 6:02:06.000 AM|
# |                null|  power off| Kippar McCaughen|             ecobee4| 5:40:19.000 PM|
# |                null|  power off|    Sidney Jotham|  GreenIQ Controller| 4:54:28.000 PM|
# |                null|  power off|    Fanya Menzies|             ecobee4| 3:12:48.000 PM|
# |                null|low battery|    Jeanne Gresch|             ecobee4| 5:39:47.000 PM|
# |                null|   power on|    Chen Cuttelar| August Doorbell Cam| 2:45:44.000 PM|
# |                null|  power off|       Merwyn Mix|         Amazon Echo| 9:23:41.000 PM|
# |                null|  power off| Angelico Conrath|         Amazon Echo| 4:53:13.000 AM|
# |                null|   power on|     Gilda Emmett| August Doorbell Cam|12:32:29.000 AM|
# |                null|low battery|  Austine Davsley|             ecobee4| 3:35:12.000 AM|
# |                null|low battery| Zackariah Thoday|         Amazon Echo| 1:26:13.000 PM|
# |                null|  power off|     Ewen Gillson|         Amazon Echo| 7:47:20.000 AM|
# |                null|   power on|     Itch Durnill|             ecobee4| 4:45:55.000 AM|
# |                null|  power off|        Winni Dow|  GreenIQ Controller| 4:12:54.000 AM|
# |                null|   power on|Talbot Valentelli| August Doorbell Cam| 7:35:23.000 PM|
# |                null|low battery|    Vikki Muckeen| August Doorbell Cam| 1:17:30.000 PM|
# |                null|  power off|  Christie Karran|Footbot Air Quali...| 9:38:13.000 PM|
# |                null|low battery|     Evonne Guest|         Amazon Echo| 8:02:21.000 AM|
# +--------------------+-----------+-----------------+--------------------+---------------+
```
A weird column is *_corrupt_record* and first value is **[{"time":"3:57:09...** in this column.
Go back to check source file and notice that it's a list of object in json file.
###### Remove  <span style='color:red'> *\[* </span> and <span style='color:red'>*\]* </span> in source file
```python
input = spark.read.json(file_path+'/1.json')

input.show()

# +-----------+-----------------+--------------------+---------------+
# |     action|         customer|              device|           time|
# +-----------+-----------------+--------------------+---------------+
# |  power off|      Alexi Barts|  GreenIQ Controller| 3:57:09.000 PM|
# |  power off|Nicolle Pargetter| August Doorbell Cam| 1:29:05.000 AM|
# |   power on|   Concordia Muck|Footbot Air Quali...| 6:02:06.000 AM|
# |  power off| Kippar McCaughen|             ecobee4| 5:40:19.000 PM|
# |  power off|    Sidney Jotham|  GreenIQ Controller| 4:54:28.000 PM|
# |  power off|    Fanya Menzies|             ecobee4| 3:12:48.000 PM|
# |low battery|    Jeanne Gresch|             ecobee4| 5:39:47.000 PM|
# |   power on|    Chen Cuttelar| August Doorbell Cam| 2:45:44.000 PM|
# |  power off|       Merwyn Mix|         Amazon Echo| 9:23:41.000 PM|
# |  power off| Angelico Conrath|         Amazon Echo| 4:53:13.000 AM|
# |   power on|     Gilda Emmett| August Doorbell Cam|12:32:29.000 AM|
# |low battery|  Austine Davsley|             ecobee4| 3:35:12.000 AM|
# |low battery| Zackariah Thoday|         Amazon Echo| 1:26:13.000 PM|
# |  power off|     Ewen Gillson|         Amazon Echo| 7:47:20.000 AM|
# |   power on|     Itch Durnill|             ecobee4| 4:45:55.000 AM|
# |  power off|        Winni Dow|  GreenIQ Controller| 4:12:54.000 AM|
# |   power on|Talbot Valentelli| August Doorbell Cam| 7:35:23.000 PM|
# |low battery|    Vikki Muckeen| August Doorbell Cam| 1:17:30.000 PM|
# |  power off|  Christie Karran|Footbot Air Quali...| 9:38:13.000 PM|
# |low battery|     Evonne Guest|         Amazon Echo| 8:02:21.000 AM|
# +-----------+-----------------+--------------------+---------------+
```
Woo, the dataframe is correct. Let's check schema
```python
input.printSchema()
# root
#  |-- action: string (nullable = true)
#  |-- customer: string (nullable = true)
#  |-- device: string (nullable = true)
#  |-- time: string (nullable = true)
```
So far I manually modify source file and drop external schema to obtain a corret dataframe. Is there anyway to
read these files without these steps.

###### add one feature <span style='color:blue'>multiLine</span>
Read the file without schema but add one feature **multiLine**

```python
input = spark.read.json("file:///path/pyspark_test_data", multiLine=True)

# OR input = spark.read.option('multiLine', True).json("file:///path/pyspark_test_data")

# +-----------+--------------------+--------------------+---------------+
# |     action|            customer|              device|           time|
# +-----------+--------------------+--------------------+---------------+
# |   power on|     Raynor Blaskett|Nest T3021US Ther...| 3:35:09.000 AM|
# |   power on|Stafford Blakebrough|  GreenIQ Controller|10:59:46.000 AM|
# |   power on|      Alex Woolcocks|Nest T3021US Ther...| 6:26:36.000 PM|
# |   power on|      Clarice Nayshe|Footbot Air Quali...| 4:46:28.000 AM|
# |  power off|      Killie Pirozzi|Footbot Air Quali...| 8:58:43.000 AM|
# |   power on|    Lynne Dymidowicz|Footbot Air Quali...| 4:20:49.000 PM|
# |   power on|       Shaina Dowyer|             ecobee4| 3:41:33.000 AM|
# |low battery|       Barbee Melato| August Doorbell Cam|10:40:24.000 PM|
# |  power off|        Clem Westcot|Nest T3021US Ther...|11:13:38.000 PM|
# |  power off|       Kerri Galfour|         Amazon Echo|10:12:15.000 PM|
# |low battery|        Trev Ashmore|  GreenIQ Controller|11:04:41.000 AM|
# |   power on|      Coral Jahnisch| August Doorbell Cam| 3:06:31.000 AM|
# |   power on|      Feliza Cowdrey|Nest T3021US Ther...| 2:49:02.000 AM|
# |  power off|   Amabelle De Haven|Footbot Air Quali...|12:11:59.000 PM|
# |  power off|     Benton Redbourn|Nest T3021US Ther...| 3:57:39.000 AM|
# |low battery|        Asher Potten| August Doorbell Cam| 1:34:44.000 AM|
# |low battery|    Lorianne Hullyer| August Doorbell Cam| 7:26:42.000 PM|
# |  power off|     Ruperto Aldcorn|Footbot Air Quali...| 3:54:49.000 AM|
# |   power on|   Agatha Di Giacomo|Footbot Air Quali...| 7:15:20.000 AM|
# |   power on|    Eunice Penwright|             ecobee4|11:14:14.000 PM|
# +-----------+--------------------+--------------------+---------------+

input.printSchema()

# root
#  |-- action: string (nullable = true)
#  |-- customer: string (nullable = true)
#  |-- device: string (nullable = true)
#  |-- time: string (nullable = true)
```

#### change the schema
Set time as *StringType*
```python
json_schema = StructType([
    StructField("time", StringType(), True),
    StructField("customer", StringType(), True),
    StructField("action", StringType(), True),
    StructField("device", StringType(), True)
])


input = spark.read.schema(json_schema).json("file:///path/pyspark_test_data", multiLine=True)

input.show()

# +---------------+--------------------+-----------+--------------------+
# |           time|            customer|     action|              device|
# +---------------+--------------------+-----------+--------------------+
# | 3:35:09.000 AM|     Raynor Blaskett|   power on|Nest T3021US Ther...|
# |10:59:46.000 AM|Stafford Blakebrough|   power on|  GreenIQ Controller|
# | 6:26:36.000 PM|      Alex Woolcocks|   power on|Nest T3021US Ther...|
# | 4:46:28.000 AM|      Clarice Nayshe|   power on|Footbot Air Quali...|
# | 8:58:43.000 AM|      Killie Pirozzi|  power off|Footbot Air Quali...|
# | 4:20:49.000 PM|    Lynne Dymidowicz|   power on|Footbot Air Quali...|
# | 3:41:33.000 AM|       Shaina Dowyer|   power on|             ecobee4|
# |10:40:24.000 PM|       Barbee Melato|low battery| August Doorbell Cam|
# |11:13:38.000 PM|        Clem Westcot|  power off|Nest T3021US Ther...|
# |10:12:15.000 PM|       Kerri Galfour|  power off|         Amazon Echo|
# |11:04:41.000 AM|        Trev Ashmore|low battery|  GreenIQ Controller|
# | 3:06:31.000 AM|      Coral Jahnisch|   power on| August Doorbell Cam|
# | 2:49:02.000 AM|      Feliza Cowdrey|   power on|Nest T3021US Ther...|
# |12:11:59.000 PM|   Amabelle De Haven|  power off|Footbot Air Quali...|
# | 3:57:39.000 AM|     Benton Redbourn|  power off|Nest T3021US Ther...|
# | 1:34:44.000 AM|        Asher Potten|low battery| August Doorbell Cam|
# | 7:26:42.000 PM|    Lorianne Hullyer|low battery| August Doorbell Cam|
# | 3:54:49.000 AM|     Ruperto Aldcorn|  power off|Footbot Air Quali...|
# | 7:15:20.000 AM|   Agatha Di Giacomo|   power on|Footbot Air Quali...|
# |11:14:14.000 PM|    Eunice Penwright|   power on|             ecobee4|
# +---------------+--------------------+-----------+--------------------+

```
Pyspark can load json files successfully without TimestampType. However, how to handle timestamp issue in this job?

#### TimestampType
In offical document, the class *pyspark.sql.DataFrameReader* has one parameter
- timestampFormat 
> sets the string that indicates a timestamp format. 
>
> Custom date formats follow the formats at java.text.SimpleDateFormat. 
> 
> This applies to timestamp type. If None is set, it uses the default value, yyyy-MM-dd'T'HH:mm:ss.SSSXXX.

```python
input = spark.read.schema(schema).option("multiLine", True).json("file:///path/pyspark_test_data", timestampFormat="h:mm:ss.SSS aa")

input.show()
# +-------------------+--------------------+-----------+--------------------+
# |               time|            customer|     action|              device|
# +-------------------+--------------------+-----------+--------------------+
# |1970-01-01 03:35:09|     Raynor Blaskett|   power on|Nest T3021US Ther...|
# |1970-01-01 10:59:46|Stafford Blakebrough|   power on|  GreenIQ Controller|
# |1970-01-01 18:26:36|      Alex Woolcocks|   power on|Nest T3021US Ther...|
# |1970-01-01 04:46:28|      Clarice Nayshe|   power on|Footbot Air Quali...|
# |1970-01-01 08:58:43|      Killie Pirozzi|  power off|Footbot Air Quali...|
# |1970-01-01 16:20:49|    Lynne Dymidowicz|   power on|Footbot Air Quali...|
# |1970-01-01 03:41:33|       Shaina Dowyer|   power on|             ecobee4|
# |1970-01-01 22:40:24|       Barbee Melato|low battery| August Doorbell Cam|
# |1970-01-01 23:13:38|        Clem Westcot|  power off|Nest T3021US Ther...|
# |1970-01-01 22:12:15|       Kerri Galfour|  power off|         Amazon Echo|
# |1970-01-01 11:04:41|        Trev Ashmore|low battery|  GreenIQ Controller|
# |1970-01-01 03:06:31|      Coral Jahnisch|   power on| August Doorbell Cam|
# |1970-01-01 02:49:02|      Feliza Cowdrey|   power on|Nest T3021US Ther...|
# |1970-01-01 12:11:59|   Amabelle De Haven|  power off|Footbot Air Quali...|
# |1970-01-01 03:57:39|     Benton Redbourn|  power off|Nest T3021US Ther...|
# |1970-01-01 01:34:44|        Asher Potten|low battery| August Doorbell Cam|
# |1970-01-01 19:26:42|    Lorianne Hullyer|low battery| August Doorbell Cam|
# |1970-01-01 03:54:49|     Ruperto Aldcorn|  power off|Footbot Air Quali...|
# |1970-01-01 07:15:20|   Agatha Di Giacomo|   power on|Footbot Air Quali...|
# |1970-01-01 23:14:14|    Eunice Penwright|   power on|             ecobee4|
# +-------------------+--------------------+-----------+--------------------+
```

All yyyy-MM-dd are 1970-01-01 because source file only hh-mm-ss. 
These source files are in wrong format in Windows.

## Streaming Our Data

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType, StringType, StructType, StructField


spark = SparkSession.builder.appName("Test Streaming").enableHiveSupport().getOrCreate()

json_schema = StructType([
    StructField("time", StringType(), True),
    StructField("customer", StringType(), True),
    StructField("action", StringType(), True),
    StructField("device", StringType(), True)
])

streamingDF = spark.readStream.schema(json_schema) \
              .option("maxFilesPerTrigger", 1) \
              .option("multiLine", True) \
              .json("file:///path/pyspark_test_data")

streamingActionCountsDF = streamingDF.groupBy('action').count()
# streamingActionCountsDF.isStreaming
spark.conf.set("spark.sql.shuffle.partitions", "2")


# View stream in real-time
# query = streamingActionCountsDF.writeStream \
#         .format("memory").queryName("counts").outputMode("complete").start()

# format choice:
# parquet
# kafka
# console
# memory

# query = streamingActionCountsDF.writeStream \
#         .format("console").queryName("counts").outputMode("complete").start()

query = streamingActionCountsDF.writeStream.format("console") \
        .queryName("counts").outputMode("complete").start().awaitTermination(timeout=10)
# Output Mode choice:
# append
# complete
# update

```







 


