import ast
import pandas as pd
from typing import Any, List

def safe_eval(x: Any) -> List[str]:
    """
    Safely evaluate string representations of lists
    
    Args:
        x: Input value to evaluate
        
    Returns:
        List of strings
    """
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        if isinstance(x, str):
            x = x.strip('[]')
            if ',' in x:
                return [item.strip().strip("'\"") for item in x.split(',')]
            return [x.strip().strip("'\"")]
        return [str(x)]

def parse_list_input(input_str: str) -> List[str]:
    """
    Parse comma-separated string input into list
    
    Args:
        input_str: Comma-separated string
        
    Returns:
        List of strings
    """
    return [x.strip() for x in input_str.split(',')]