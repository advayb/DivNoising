#!/usr/bin/env python
import json
import os

def fix_imports_in_notebook(notebook_path):
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            for i, line in enumerate(cell['source']):
                # Fix sys.path.append
                if "sys.path.append('../../')" in line:
                    cell['source'][i] = line.replace(
                        "sys.path.append('../../')", 
                        "# Add the parent directory and model directory to the path\nsys.path.append('../')\nsys.path.append('../model')"
                    )
                
                # Fix divnoising imports
                if "from divnoising import " in line:
                    cell['source'][i] = line.replace(
                        "from divnoising import ", 
                        "# Import from local model directory instead of divnoising\nfrom model import "
                    )
                
                if "from divnoising." in line:
                    cell['source'][i] = line.replace(
                        "from divnoising.", 
                        "from model."
                    )
    
    # Write back to file
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Fixed imports in {notebook_path}")

# Fix imports in all notebook files
notebook_dir = "notebook"
for filename in os.listdir(notebook_dir):
    if filename.endswith(".ipynb"):
        fix_imports_in_notebook(os.path.join(notebook_dir, filename)) 