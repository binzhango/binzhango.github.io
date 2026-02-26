---
title: Spark SQL

date: 
  created: 2020-02-21 14:10:36

tags: 
  - Data Engineer
categories: 
  - Spark
---

<!-- more -->
## Spark Submit options

```txt
--master MASTER_URL	--> 运行模式
	例：spark://host:port, mesos://host:port, yarn, or local.
	
--deploy-mode DEPLOY_MODE 
	Whether to launch the driver program locally ("client") or
	on one of the worker machines inside the cluster ("cluster")
	(Default: client).
	
--class CLASS_NAME	--> 运行程序的main_class 
	例： com.jhon.hy.main.Test
	
--name NAME -->application 的名字
	例：my_application

--jars JARS  -->逗号分隔的本地jar包，包含在driver和executor的classpath下
	例：/home/app/python_app/python_v2/jars/datanucleus-api-jdo-3.2.6.jar,/home/app/python_app/python_v2/jars/datanucleus-core-3.2.10.jar,/home/app/python_app/python_v2/jars/datanucleus-rdbms-3.2.9.jar,/home/app/python_app/python_v2/jars/mysql-connector-java-5.1.37-bin.jar,/home/app/python_app/python_v2/jars/hive-bonc-plugin-2.0.0.jar\

--exclude-packages -->用逗号分隔的”groupId:artifactId”列表

--repositories  -->逗号分隔的远程仓库

--py-files PY_FILES  -->逗号分隔的”.zip”,”.egg”或者“.py”文件，这些文件放在python app的

--files FILES    -->逗号分隔的文件，这些文件放在每个executor的工作目录下面，涉及到的k-vge格式的参数，用 ‘#’ 连接,如果有自定义的log4j 配置，也放在此配置下面
	例：/home/app/python_app/python_v2/jars/kafka_producer.jar,/home/app/python_app/python_v2/resources/....cn.keytab#.....keytab,/home/app/python_app/python_v2/resources/app.conf#app.conf,/home/app/python_app/python_v2/resources/hive-site.xml,/home/app/python_app/python_v2/resources/kafka_client_jaas.conf#kafka_client_jaas.conf

--properties-file FILE   --> 默认的spark配置项，默认路径 conf/spark-defaults.conf

--conf PROP=VALUE  -->	任意的spark配置项
 	例： --conf "spark.driver.maxResultSize=4g" \
		--conf spark.sql.shuffle.partitions=2600 \
		--conf spark.default.parallelism=300 \

--driver-memory MEM         Memory for driver (e.g. 1000M, 2G) (Default: 1024M).
--driver-java-options       Extra Java options to pass to the driver.
--driver-library-path       Extra library path entries to pass to the driver.
--driver-class-path         Extra class path entries to pass to the driver. Note that
                            jars added with --jars are automatically included in the
                            classpath.

--executor-memory MEM       Memory per executor (e.g. 1000M, 2G) (Default: 1G).

--proxy-user NAME           User to impersonate when submitting the application.
                            This argument does not work with --principal / --keytab.

--help, -h                  Show this help message and exit.
--verbose, -v               Print additional debug output.
--version,                  Print the version of current Spark.

 Spark standalone with cluster deploy mode only:
  --driver-cores NUM          Cores for driver (Default: 1).

 Spark standalone or Mesos with cluster deploy mode only:
  --supervise                 If given, restarts the driver on failure.
  --kill SUBMISSION_ID        If given, kills the driver specified.
  --status SUBMISSION_ID      If given, requests the status of the driver specified.

 Spark standalone and Mesos only:
  --total-executor-cores NUM  Total cores for all executors.

 Spark standalone and YARN only:
  --executor-cores NUM        Number of cores per executor. (Default: 1 in YARN mode,
                              or all available cores on the worker in standalone mode)

 YARN-only:
  --driver-cores NUM          Number of cores used by the driver, only in cluster mode
                              (Default: 1).
  --queue QUEUE_NAME          The YARN queue to submit to (Default: "default").
  --num-executors NUM         Number of executors to launch (Default: 2).
                              If dynamic allocation is enabled, the initial number of
                              executors will be at least NUM.
  --archives ARCHIVES         Comma separated list of archives to be extracted into the
                              working directory of each executor.
  --principal PRINCIPAL       Principal to be used to login to KDC, while running on
                              secure HDFS.
  --keytab KEYTAB             The full path to the file that contains the keytab for the
                              principal specified above. This keytab will be copied to
                              the node running the Application Master via the Secure
                              Distributed Cache, for renewing the login tickets and the
                              delegation tokens periodically.



--num-executors 30 \ 	启动的executor数量。默认为2。在yarn下使用
--executor-cores 4 \ 	每个executor的核数。在yarn或者standalone下使用
--driver-memory 8g \ 	Driver内存，默认1G
--executor-memory 16g	每个executor的内存，默认是1G

`通常我们讲用了多少资源是指: num-executor * executor-cores 核心数，--num-executors*--executor-memory 内存`

`因现在所在公司用的是spark on yarn 模式，以下涉及调优主要针对目前所用`

--num-executors 这个参数决定了你的程序会启动多少个Executor进程来执行，YARN集群管理
				器会尽可能按照你的设置来在集群的各个工作节点上，启动相应数量的Executor
				进程。如果忘记设置，默认启动两个，这样你后面申请的资源再多，你的Spark程
				序执行速度也是很慢的。
	调优建议： 这个要根据你程序运行情况，以及多次执行的结论进行调优，太多，无法充分利用资
			 源，太少，则保证不了效率。

--executor-memory   这个参数用于设置每个Executor进程的内存，Executor的内存很多时候决
					定了Spark作业的性能，而且跟常见的JVM OOM也有直接联系
	调优建议：参考值 --> 4~8G,避免程序将整个集群的资源全部占用,需要先看一下你队列的最大
			内存限制是多少，如果是公用一个队列，你的num-executors * executor-memory
			最好不要超过队列的1/3 ~ 1/2
-- executor-cores
参数说明：
	该参数用于设置每个Executor进程的CPU core数量。这个参数决定了每个Executor进程并行执行task线程的能力。因为每个CPU core同一时间只能执行一个
	task线程，因此每个Executor进程的CPU core数量越多，越能够快速地执行完分配给自己的所有task线程。
参数调优建议：
	Executor的CPU core数量设置为2~4个较为合适。同样得根据不同部门的资源队列来定，可以看看自己的资源队列的最大CPU core限制是多少，再依据设置的
	Executor数量，来决定每个Executor进程可以分配到几个CPU core。同样建议，如果是跟他人共享这个队列，那么num-executors * executor-cores不要超过
	队列总CPU core的1/3~1/2左右比较合适，也是避免影响其他同学的作业运行。

--driver-memory
参数说明：
	该参数用于设置Driver进程的内存。
参数调优建议：
	Driver的内存通常来说不设置，或者设置1G左右应该就够了。唯一需要注意的一点是，如果需要使用collect算子将RDD的数据全部拉取到Driver上进行处理，
	那么必须确保Driver的内存足够大，否则会出现OOM内存溢出的问题。

--spark.default.parallelism
参数说明：
	该参数用于设置每个stage的默认task数量。这个参数极为重要，如果不设置可能会直接影响你的Spark作业性能。
参数调优建议：
	Spark作业的默认task数量为500~1000个较为合适。很多同学常犯的一个错误就是不去设置这个参数，那么此时就会导致Spark自己根据底层HDFS的block数量
	来设置task的数量，默认是一个HDFS block对应一个task。通常来说，Spark默认设置的数量是偏少的（比如就几十个task），如果task数量偏少的话，就会
	导致你前面设置好的Executor的参数都前功尽弃。试想一下，无论你的Executor进程有多少个，内存和CPU有多大，但是task只有1个或者10个，那么90%的
	Executor进程可能根本就没有task执行，也就是白白浪费了资源！因此Spark官网建议的设置原则是，设置该参数为num-executors * executor-cores的2~3倍
	较为合适，比如Executor的总CPU core数量为300个，那么设置1000个task是可以的，此时可以充分地利用Spark集群的资源。

--spark.storage.memoryFraction
参数说明：
	该参数用于设置RDD持久化数据在Executor内存中能占的比例，默认是0.6。也就是说，默认Executor 60%的内存，可以用来保存持久化的RDD数据。根据你选择
	的不同的持久化策略，如果内存不够时，可能数据就不会持久化，或者数据会写入磁盘。
参数调优建议：
	如果Spark作业中，有较多的RDD持久化操作，该参数的值可以适当提高一些，保证持久化的数据能够容纳在内存中。避免内存不够缓存所有的数据，导致数据只
	能写入磁盘中，降低了性能。但是如果Spark作业中的shuffle类操作比较多，而持久化操作比较少，那么这个参数的值适当降低一些比较合适。此外，如果发现
	作业由于频繁的gc导致运行缓慢（通过spark web ui可以观察到作业的gc耗时），意味着task执行用户代码的内存不够用，那么同样建议调低这个参数的值。

--spark.shuffle.memoryFraction
参数说明：
	该参数用于设置shuffle过程中一个task拉取到上个stage的task的输出后，进行聚合操作时能够使用的Executor内存的比例，默认是0.2。也就是说，Executor
	默认只有20%的内存用来进行该操作。shuffle操作在进行聚合时，如果发现使用的内存超出了这个20%的限制，那么多余的数据就会溢写到磁盘文件中去，此时
	就会极大地降低性能。
参数调优建议：
	如果Spark作业中的RDD持久化操作较少，shuffle操作较多时，建议降低持久化操作的内存占比，提高shuffle操作的内存占比比例，避免shuffle过程中数据过多
	时内存不够用，必须溢写到磁盘上，降低了性能。此外，如果发现作业由于频繁的gc导致运行缓慢，意味着task执行用户代码的内存不够用，那么同样建议调低
	这个参数的值。
资源参数的调优，没有一个固定的值，需要根据自己的实际情况（包括Spark作业中的shuffle操作数量、RDD持久化操作数量以及spark web ui中显示的作业gc情况），
合理地设置上述参				
```