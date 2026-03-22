"""
Example utility functions

This module contains example utility functions that demonstrate
how to add custom functions to this package.
"""


def greet(name):
    """
    Simple greeting function as an example.
    
    Args:
        name (str): The name to greet
        
    Returns:
        str: A greeting message
        
    Example:
        >>> from general_utils.example_utils import greet
        >>> greet("World")
        'Hello, World!'
    """
    return f"Hello, {name}!"


def add_numbers(a, b):
    """
    Add two numbers together.
    
    Args:
        a (int/float): First number
        b (int/float): Second number
        
    Returns:
        int/float: Sum of a and b
        
    Example:
        >>> from general_utils.example_utils import add_numbers
        >>> add_numbers(2, 3)
        5
    """
    return a + b
