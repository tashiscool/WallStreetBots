#!/usr/bin/env python3
"""
Script to fix common linting issues in the tradingbot module
"""

import os
import re
import glob

def fix_file(filepath):
    """Fix linting issues in a single file"""
    print(f"Fixing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix unused imports
    unused_imports = [
        'from dataclasses import dataclass, field',
        'from typing import Tuple',
        'from typing import Union', 
        'from typing import Optional',
        'from typing import Dict',
        'from typing import List',
        'import math',
        'import json',
        'import pandas as pd',
        'import numpy as np',
        'from datetime import timedelta',
        'from datetime import timezone',
        'from typing import Any',
        'from .risk_management import PositionStatus',
        'from .risk_management import RiskParameters',
        'from .options_calculator import OptionsSetup',
        'from .market_regime import TechnicalIndicators',
        'from .risk_management import PositionSizer',
        'from dataclasses import dataclass, asdict',
        'from typing import List',
        'from typing import Dict',
        'from typing import Optional',
    ]
    
    for unused_import in unused_imports:
        # Remove unused imports
        if unused_import in content:
            content = content.replace(unused_import + '\n', '')
    
    # Fix f-strings without placeholders
    f_string_pattern = r'f"([^"]*)"'
    matches = re.findall(f_string_pattern, content)
    for match in matches:
        if '{' not in match and '}' not in match:
            old_fstring = f'f"{match}"'
            new_string = f'"{match}"'
            content = content.replace(old_fstring, new_string)
    
    # Fix bare except clauses
    content = re.sub(r'except:\s*$', 'except Exception:', content, flags=re.MULTILINE)
    
    # Fix trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Ensure file ends with newline
    if content and not content.endswith('\n'):
        content += '\n'
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Fixed {filepath}")
    else:
        print(f"  No changes needed for {filepath}")

def main():
    """Fix all Python files in backend/tradingbot/"""
    python_files = glob.glob('backend/tradingbot/*.py')
    
    for filepath in python_files:
        fix_file(filepath)
    
    print("Linting fixes complete!")

if __name__ == '__main__':
    main()
