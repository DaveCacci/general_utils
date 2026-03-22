# Contributing to general_utils

Thank you for considering contributing to general_utils! This document provides guidelines for adding new utility functions to the package.

## Adding New Utility Functions

1. **Create a new module** (optional if adding to existing module):
   ```bash
   cd general_utils
   touch your_module_name.py
   ```

2. **Write your utility function with proper documentation**:
   ```python
   def your_function(arg1, arg2):
       """
       Brief description of what the function does.
       
       Args:
           arg1 (type): Description of arg1
           arg2 (type): Description of arg2
           
       Returns:
           type: Description of return value
           
       Example:
           >>> from general_utils.your_module_name import your_function
           >>> your_function(1, 2)
           3
       """
       # Your implementation here
       return result
   ```

3. **Optionally expose the function in `__init__.py`**:
   ```python
   from .your_module_name import your_function
   ```

4. **Test your function**:
   ```bash
   python3 -c "from general_utils.your_module_name import your_function; print(your_function(test_args))"
   ```

5. **Commit and push your changes**:
   ```bash
   git add .
   git commit -m "Add your_function to your_module_name"
   git push
   ```

## Best Practices

- Write clear, descriptive docstrings for all functions
- Include type hints where appropriate
- Add usage examples in docstrings
- Keep functions focused and single-purpose
- Use meaningful variable and function names
- Follow PEP 8 style guidelines

## Code Style

- Use 4 spaces for indentation
- Keep lines under 100 characters when possible
- Use descriptive variable names
- Add comments for complex logic

## Questions?

If you have questions or need help, please open an issue on GitHub.
