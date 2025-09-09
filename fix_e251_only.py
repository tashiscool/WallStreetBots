#!/usr/bin/env python3
"""
Fix only E251 issues (unexpected spaces around keyword / parameter equals)
"""

import os
import re
import glob

def fix_e251_only(filepath):
    """Fix only E251 issues in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content=f.read()
        
        original_content=content
        
        # Fix E251: unexpected spaces around keyword / parameter equals
        # This is the most common issue - remove spaces around = in function calls
        # But be careful not to break assignment statements
        
        # Pattern 1: function calls with keyword arguments
        # e.g., function(param=value, other=value) -> function(param=value, other=value)
        content=re.sub(r'(\w+)\s*=\s*([^,)]+)', r'\1=\2', content)
        
        # Pattern 2: dictionary definitions
        # e.g., {"key":value, "other":value} -> {"key":value, "other":value}
        content=re.sub(r'"([^"]+)"\s*:\s*', r'"\1":', content)
        content=re.sub(r"'([^']+)'\s*:\s*", r"'\1':", content)
        
        # Pattern 3: function definitions with default parameters
        # e.g., def func(param=default, other=default) -> def func(param=default, other=default)
        content=re.sub(r'def\s+(\w+)\s*\(([^)]*)\)', 
                        lambda m: f"def {m.group(1)}({re.sub(r'(\w+)\s*=\s*([^,)]+)', r'\1=\2', m.group(2))})", 
                        content)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed E251 in: {filepath}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Fix E251 issues in all Python files"""
    python_files=[]
    
    # Get all Python files
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'test_models']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    
    fixed_count=0
    for filepath in python_files:
        if fix_e251_only(filepath):
            fixed_count += 1
    
    print(f"Fixed E251 in {fixed_count} files")

if __name__== "__main__":main()
