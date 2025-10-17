#!/usr/bin/env python3
"""
Clean Jupyter notebooks - remove kernel info and execution counts.

Usage:
    python clean_notebooks.py
"""

import json
from pathlib import Path


def clean_notebook(notebook_path: Path) -> None:
    """Clean a single notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Remove kernel metadata
    if 'metadata' in nb:
        # Keep minimal metadata
        nb['metadata'] = {
            'language_info': {
                'name': 'python',
                'version': '3.9.0'
            }
        }

    # Clean cells
    if 'cells' in nb:
        for cell in nb['cells']:
            # Remove execution count
            if 'execution_count' in cell:
                cell['execution_count'] = None

            # Clear outputs for code cells
            if cell.get('cell_type') == 'code' and 'outputs' in cell:
                cell['outputs'] = []

            # Clean cell metadata
            if 'metadata' in cell:
                cell['metadata'] = {}

    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"✓ Cleaned {notebook_path.name}")


def main():
    """Clean all notebooks in examples/jupyter/"""
    jupyter_dir = Path('examples/jupyter')

    if not jupyter_dir.exists():
        print(f"Directory {jupyter_dir} not found")
        return

    notebooks = list(jupyter_dir.glob('*.ipynb'))

    if not notebooks:
        print("No notebooks found")
        return

    print(f"Cleaning {len(notebooks)} notebooks...")

    for nb_path in notebooks:
        clean_notebook(nb_path)

    print(f"\n✓ Cleaned {len(notebooks)} notebooks successfully!")


if __name__ == '__main__':
    main()
