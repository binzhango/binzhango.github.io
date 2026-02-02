---
title: Python Decorator
authors:
  - BZ
date: 2024-10-23
categories: 
  - python
---

# Python decorators
:question_mark: Why we need decorator

:bulb: It will extend your function behaviors during runtime.

<!-- more -->
For example, you already have a function `say_hi`

```python linenums="1"
def say_hi(name: str):
    return f"Hi! {name}"
```

- function name `say_hi`
- parameter `name` and type `str`
- one output `say_hi('Apple') => Hi! Apple`

Next we plan to add one introduction to the output, such as `Hi! Apple. I'm Bob`.

1. Modify your function
```python linenums="1"
def say_hi(name: str, my_name:str):
    return f"Hi! {name}. I'm {my_name}"
```
If this is only used once in our project, it’s manageable. 
However, modifying the function signature means that every instance of its use throughout the project must be updated, which is both tedious and time-consuming.

2. Use decorator
```python linenums="1"
def add_intro(my_name):
    def dec(func):
        def wrapper(name):
            return func(name) + f". I'm {my_name}"
        return wrapper
    return dec

@add_intro("Bob")
def say_hi(name: str):
    return f"Hi! {name}"
```
:exclamation:`Function signature is not changed and function behavior is enriched`

# How to create decorator

## Decorator Function
Before starting decorator, we have to understand

- original function
    * function name
    * function parameters and types
- decorator function
    * extra parameter
    * new features

```python linenums="1" title="original function"
def hello(name:str) -> str:
    return f"hello, {name}"
```

> We have an original function
>
> - function name: `hello`
> - parameters: `name`
> - types: `str`

Now we can use these to build decorator function `my_dec`
```python title="my_dec" linenums="1"
def my_dec(func):
    def wrapper(name:str):
        return func(name)
    return wrapper
```

**Explanation:**

- `line 1`
    * decorator name : `my_dec`
    * decorator parameter : `func` (:warning: we did not use type hint here)
    * it mean `my_dec` will decorate function `func`
- `line 2`
    * inner function `wrapper` (:warning: any function names)
    * inner function **signature**. It MUST be a superset of your original signature
    > e.g.
    > only 1 parameter `name` at `hello`
    > the wrapper function should include `name` at least
    > it could be `wrapper(name)`, `wrapper(name, name1=None)`, `wrapper(name, *args, **kwargs)`
    > `wrapper(*args, **kwargs)` etc.
- `line 3`
    * `return value`
- `line 4`
    * :exclamation: return a function name `wrapper`

Now the decorator is working as `func = my_dec(func)`

- function **IN** : `func`
- function **OUT**: `wrapper` and reassigned `wrapper` to `func`


## Decorator Class

```python title="Decorator Class" linenums="1"
class DecoratorClass:
    def __init__(self, decorator_param: str):
        self.decorator_param = decorator_param
        
    def __call__(self, func):
        def wrapper(original_param):
            """wrapper doc"""
            return func(original_param) + self.decorator_param
        return wrapper

@DecoratorClass(decorator_param="!!!")
def hello_again(name: str):
    """original docStr"""
    return f"Hello {name}"
```

it's obviously to understand how to setup decorator's parameters.

## **built-in** python decorator
Each function in `python` has metadata

- `__name__`
- `__doc__`

Either `function` or `class` cannot update decorator function metadata.

```python 
print(hello_again.__name__) # wrapper

print(hello_again.__doc__) # Wrapper Doc
```

:question_mark: **How to update function metadata**

- manually update
    ```python
    class DecoratorClass1:
    def __init__(self, decorator_param: str):
        self.decorator_param = decorator_param

    def __call__(self, func):
        def wrapper(original_param):
            """Wrapper Doc"""
            return func(original_param) + self.decorator_param

        # Manually update metadata
        wrapper.__doc__ = func.__doc__
        wrapper.__name__ = func.__name__

        return wrapper
    ```
- use python built-in `wrapper`


