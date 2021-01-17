# Initialization
## Clone project
```
git clone https://github.com/yari61/otus-patterns-multiprocess.git
cd otus-patterns-multiprocess
```

## Virtual environment
It is recommended to create a virtual environment at first (.venv for example)
```
python -m venv .venv
```

Then activate it with 
- ```source .venv/bin/activate```
on Unix-like systems, or
- ```.venv\bin\activate```
if Your system runs Windows

## Installation
To install the package run the next command in your virtual environment
```
pip install -e .
```

## Testing
To run all tests execute this
```
python -m unittest
```
To run unit tests only
```
python -m unittest discover tests.unit
```
To run functional tests only
```
python -m unittest discover tests.functional
```

# Project description
## Class diagram
![Alt](docs/images/uml_class_diagram.png)
