# general_utils

Common custom functions across projects

## Description

This repository contains reusable Python utility functions that can be shared across multiple projects. It is designed to be installed directly from GitHub and used as a Python package.

## Installation

You can install this package directly from GitHub using pip:

```bash
pip install git+https://github.com/DaveCacci/general_utils.git
```

Or if you want to install a specific branch:

```bash
pip install git+https://github.com/DaveCacci/general_utils.git@branch-name
```

For development installation (editable mode):

```bash
git clone https://github.com/DaveCacci/general_utils.git
cd general_utils
pip install -e .
```

## Usage

After installation, you can import and use the utility functions in your projects:

```python
from general_utils.example_utils import greet, add_numbers

# Use the functions
print(greet("World"))  # Output: Hello, World!
print(add_numbers(2, 3))  # Output: 5
```

## Adding New Utilities

To add new utility functions to this package:

1. Create a new Python module in the `general_utils/` directory (e.g., `string_utils.py`)
2. Define your functions with proper docstrings
3. Optionally, import them in `general_utils/__init__.py` for easier access
4. Commit and push your changes

## Structure

```
general_utils/
├── README.md
├── setup.py
├── .gitignore
└── general_utils/
    ├── __init__.py
    └── example_utils.py
```

## Requirements

- Python >= 3.6

## License

This project is licensed under the MIT License.
