---
title: Spark Optimization

date: 
  created: 2020-02-21 14:10:36

tags: 
  - Data Engineer
categories: 
  - Spark
---

# Spark run faster and faster
- Cluster Optimization
- Parameters Optimization
- Code Optimization
<!-- more -->

## Cluster Optimization

#### Locality Level
Data locality is how close data is to the code processing it. There are several levels of locality based on the data’s current location. In order from closest to farthest:

- **PROCESS_LOCAL** data is in the same JVM as the running code. This is the best locality possible
- **NODE_LOCAL** data is on the same node. Examples might be in HDFS on the same node, or in another executor on the same node. This is a little slower than PROCESS_LOCAL because the data has to travel between processes
- **NO_PREF** data is accessed equally quickly from anywhere and has no locality preference
- **RACK_LOCAL** data is on the same rack of servers. Data is on a different server on the same rack so needs to be sent over the network, typically through a single switch
- **ANY** data is elsewhere on the network and not in the same rack

Performance: PROCESS_LOCAL > NODE_LOCAL > NO_PREF > RACK_LOCAL

###### Locality settting
- spark.locality.wait.process
- spark.locality.wait.node
- spark.locality.wait.rack

#### Data Format
- text
- orc
- parquet
- avro
###### format setting
- spark.sql.hive.convertCTAS
- spark.sql.sources.default


#### parallelising
- spark.sql.shuffle.partitions : default is 200

#### computing
- --executor-memory : default is 1G
- --executor-cores : default is 1
if large memory cause resource throtle in cluster, if small memory cause task termination
if more cores cause IO issue, if less cores slow dow computing

#### memory
- spark.executor.overhead.memory

#### table join
- spark.sql.autoBroadcastJoinThreshold : default 10M

#### predicate push down in Spark SQL queries
- spark.sql.parquet.filterPushdown : default True
- spark.sql.orc.filterPushdown=true : default False

#### reuse RDD
```pytthon
    df.persist(pyspark.StorageLevel.MEMORY_ONLY)
```

#### Spark operators
- shuffle operators
  - avoid using <span style="color:blue"> **reduceByKey**, **join**, **distinct**, **repartition** etc</span>
  - Broadcast small dataset

- High performance operator
  - reduceByKey > groupByKey (reduceByKey works at map side)
  - mapPartitions > map (reduce function calls)
  - treeReduce > reduce (treeReduce works at executor not driver)
    - treeReduce & reduce return some result to driver
    - treeReduce does more work on the executors while reduce bring everything back to the driver.
  - foreachPartitions > foreach (reduce function calls)
  - filter -> coalesce (reduce number of partitions and reduce tasks)
  - repartitionAndSortWithinPartitions > repartition & sort
  - broadcast (100M)


#### shuffle
- spark.shuffle.sort.bypassMergeThreshold
- spark.shuffle.io.retryWait
- spark.shuffle.io.maxRetries


TBC