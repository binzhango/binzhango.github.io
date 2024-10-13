---
title: Snowflake Data Science Training Summary
date: 
  created: 2024-10-05

tags:
  - Data Science
categories: 
  - Snowflake
---

<!-- more -->

## Snowflake Data Science

I have enrolled in a private Snowflake Data Science Training.
Let me list what I learned from it.

- SQL worksheets
- Snowpark in notebook

```sh
mkdocs build
mkdocs serve
mkdocs gh-deploy --force
```

## SQL Worksheets
ML functions:

- forecast
- anomaly_detection
- classification
- top_insights

### Add object name into session
```sql
show parameters like 'SEARCH_PATH';

set cur_search_path = (select "value" from table(result_scan(-1)));
set new_search_path = (select $cur_search_path || ', snowflake.ml'); -- append `snowflake.ml` into search_path

alter session set search_path = $new_search_path;

-- now below two statements are interchangeable 
show snowflake.ml.forecast;
show forecast;
```

## Snowpark Notebook

### Snowpark Configuration & Snowflak-Spark Configuration

 :exclamation: **Attributes are different**

- Snowpark config
  ```json
  {
    "account":"",
    "user":"",
    "authenticator":"externalbrowser",
    "role":"",
    "warehouse":"",
    "database":"",
    "schema":""
  }
  ```
- Snowflake Spark config
  ```json
  {
    "sfURL":"",
    "sfRole":"",
    "sfWarehouse":"",
    "sfDatabase":"",
    "sfSchema":"",
    "sfUser":"",
    "sfPassword":"",
    "authenticator":"externalbrowser",
  }
  ```


