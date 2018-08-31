import pytest

if not pytest.config.getoption("--notebook-tests"):
    pytest.skip("--notebook-tests is missing, skipping test of examples notebook", allow_module_level=True)

"""
import subprocess
import tempfile
import os

def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["python3","-m","jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output-dir=notebooks", "--output", fout.name, path]
        subprocess.check_call(args)

def test():
	_exec_notebook('notebooks/examples.ipynb')
"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

def _exec_notebook(notebook_filename):
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})

def test_examples():
	_exec_notebook('notebooks/examples.ipynb')
