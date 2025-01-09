---
title: PySpark Dataframe Transformation
authors:
  - BZ
date: 2024-11-15
categories: 
  - python
  - spark
---

# Migration from `Scala` to `Python`
Migrating a history `Scala` project to `Python`, I find some tips that can help me
forget the `type` system in `scala`. Feel good!!! :smile:

<!-- more -->

## `dataclass` vs `case class`
You have to create a `case class` for each data model in Scala, 
while`dataclass` is your alternative in python

```python linenums="1"
@dataclass()
class Event:
    event_id: int
    event_name: str
```

### Create Dataframe from `dataclass`

```python linenums="1"
spark = (
    SparkSession.builder.master("local[*]")
    .appName("test")
    .getOrCreate()
)
d = [
    Event(1, "abc"),
    Event(2, "ddd"),
]

# Row object
df = spark.createDataFrame(Row(**e.__dict__) for e in d)
df.show()
# +--------+----------+
# |event_id|event_name|
# +--------+----------+
# |       1|       abc|
# |       2|       ddd|
# +--------+----------+

```