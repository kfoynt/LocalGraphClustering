import subprocess
import tempfile
import os


def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["python3","-m","jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000", "--ExecutePreprocessor.kernel_name=python3",
                "--output", fout.name, path]
        subprocess.check_call(args,cwd="/homes/liu1740/Research/LocalGraphClustering")


def test():
	os.system('cp notebooks/examples.ipynb .')
	_exec_notebook('examples.ipynb')
	os.system('rm examples.ipynb')