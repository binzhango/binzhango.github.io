---
title: Airflow
authors:
  - BZ
date: 
  created: 2020-02-11 22:20:40
tags:
  - Data Engineer
categories: 
  - airflow
---

<!-- more -->

## Code snippet


```py
 
import airflow
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
  
default_args = {
	'owner': 'ABC',
	'start_date': airflow.utils.dates.days_ago(1),
	'depends_on_past': False,
    # failure email
	'email': ['abc@xxx.com'],
	'email_on_failure': True,
	'email_on_retry': True,
	'retries': 3,
	'retry_delay': timedelta(minutes=5),
	'pool': 'data_hadoop_pool',
	'priority_weight': 900,
	'queue': '66.66.0.66:8080'
}
 
dag = DAG(
    dag_id='daily', 
    default_args=default_args,
    schedule_interval='0 13 * * *')

def fetch_data_from_hdfs_function(ds, **kwargs):
	pass
 
def push_data_to_mysql_function(ds, **kwargs):
	pass
 
fetch_data_from_hdfs = PythonOperator(
	task_id='fetch_data_from_hdfs',
	provide_context=True,
	python_callable=fetch_data_from_hdfs_function,
	dag=dag)
 
push_data_to_mysql = PythonOperator(
	task_id='push_data_to_mysql',
	provide_context=True,
	python_callable=push_data_to_mysql_function,
	dag=dag)
 
fetch_data_from_hdfs >> push_data_to_mysql
```

## update 
```python
#default parameters
fetch_data_from_hdfs = PythonOperator(
	task_id='fetch_data_from_hdfs',
	provide_context=True,
	python_callable=fetch_data_from_hdfs_function,
	dag=dag)
 
#overwrite parameters
push_data_to_mysql = PythonOperator(
    task_id='push_data_to_mysql',
    queue='77.66.0.66:8080', #update
    pool='data_mysql_pool', #update
    provide_context=True,
    python_callable=push_data_to_mysql_function,
    dag=dag)
```

## decouple

```py
import xx.fetch_data_from_hdfs 
 
def fetch_data_from_hdfs_function(ds, **kwargs):
	if not fetch_data_from_hdfs: 
        raise AirflowException('run fail: fetch_data_from_hdfs')
 
fetch_data_from_hdfs = PythonOperator(
	task_id='fetch_data_from_hdfs',
	provide_context=True,
	python_callable=fetch_data_from_hdfs_function,
	dag=dag)
```


