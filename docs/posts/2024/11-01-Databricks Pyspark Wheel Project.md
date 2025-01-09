---
title: Databricks Wheel Job
authors:
  - BZ
date: 2024-11-01
categories: 
  - python
---

# Databricks Jobs
<!-- more -->
Recently I successfully deploy my python wheel to Databricks Cluster.
Here are some tips if you plan to deploy `pyspark`.

- `pyspark` project
- `pytest`

## `pyspark` project

>My previous spark project is `scala` based and I use `IDEA` to `compile` and `test` conveniently.
>
>:smile::smile::smile:
>
>`Databricks` Job nice UI save your time to create `JAR` job.

This is official guide:
[Databricks Wheel Job](https://learn.microsoft.com/en-us/azure/databricks/jobs/how-to/use-python-wheels-in-workflows)

**What I did**:

1. Initialize a python project
    ```sh linenums='1'
    # create python virtual environment
    python -m venv pyspark_venv

    # active your venv
    source pyspark_venv/bin/activate

    # check your current python
    which python

    # install python lib
    pip install uv ruff pyspark pytest wheel

    ## if pip failed at proxy error
    ## adding your proxy 
    ## --proxy http://proxy:port

    # create your project
    uv init --package <your package name>
    ```

    After `uv` command complete, a nice python project is created.
    ```sh linenums='1'
    pyspark-app
    ├── README.md
    ├── pyproject.toml
    └── src
        └── pyspark_app
            └── __init__.py
    ```

2. :exclamation: pyspark `entry point`
    - add one file `__main__.py` in pyspark_app
    - modify `[project.scripts]` in `pyproject.toml` and this is `entry point` of Databricks job

    Now the project is
    ```sh linenums='1'
    pyspark-app
    ├── README.md
    ├── pyproject.toml
    └── src
        └── pyspark_app
            ├── __init__.py
            └── __main__.py
    ```


## `pytest`
Please check your `pytest` installed.
Let create a new package `test`

```sh linenums='1'
pyspark-app
├── README.md
├── pyproject.toml
└── src
    └── pyspark_app
        ├── __init__.py
        ├── __main__.py
        └── test
            ├── __init__.py
            ├── conftest.py
            └── test_spark.py
```

```py title='test_spark' linenums='1'
def test_spark(init_spark):
    spark = init_spark
    df = spark.range(10)
    df.show()

""" output
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
24/11/01 20:59:42 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
PASSED                                         [100%]+---+
| id|
+---+
|  0|
|  1|
|  2|
|  3|
|  4|
|  5|
|  6|
|  7|
|  8|
|  9|
+---+
"""
```

Now you can work on your spark application with test

## wheel file
Final step is building `wheel` file

```sh linenums='1'
# 1. change your work directory to pyproject.toml
# 2. run below command
python -m build --wheel

# project is now changing to

pyspark-app
├── README.md
├── build
│   ├── bdist.macosx-12.0-x86_64
│   └── lib
│       └── pyspark_app
│           ├── __init__.py
│           ├── __main__.py
│           └── test
│               ├── __init__.py
│               ├── conftest.py
│               └── test_spark.py
├── dist
│   └── pyspark_app-0.1.0-py3-none-any.whl
├── pyproject.toml
└── src
    ├── pyspark_app
    │   ├── __init__.py
    │   ├── __main__.py
    │   └── test
    │       ├── __init__.py
    │       ├── conftest.py
    │       └── test_spark.py
    └── pyspark_app.egg-info
        ├── PKG-INFO
        ├── SOURCES.txt
        ├── dependency_links.txt
        ├── entry_points.txt
        └── top_level.txt
```
Your wheel file is at line `20`

Go to view all at 
[Project template](https://github.com/binzhango/databricks_wheel_project.git)