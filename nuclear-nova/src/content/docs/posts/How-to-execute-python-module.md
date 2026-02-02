---
title: How to execute python modules
authors:
  - BZ
date: 2024-09-08 00:00:00
categories: 
  - python
---

<!-- more -->

# **runpy** module

We can use internal `runpy` to execute different moduls in our project.

This is used in my pyspark project.

```python title="submit.py" linenums="1"
import runpy
import sys

if __name__ == '__main__':
    module_name = sys.argv[1]
    function_name = sys.argv[2]
    sys.argv = sys.argv[2:] # this is important for next python entry point
    runpy.run_module(module_name, run_name=function_name)
```

Now, the spark job can be invoked by

```bash linenums="1"
spark-submit submit.py "<module_name>" "<function_name>"
```

Also, we can wrapper this shell command into a script.

```bash title="run.sh" linenums="1"
module_name=$1
function_name=$2
spark-submit submit.py "$module_name" "$function_name" "${@:3}"
```