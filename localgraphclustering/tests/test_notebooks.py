import subprocess
import tempfile
import os


def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000", "--ExecutePreprocessor.kernel_name=python3",
                "--output", fout.name, path]
        subprocess.check_call(args)


def test():
	os.chdir("/home/travis/build/MengLiuPurdue/LocalGraphClustering")
    _exec_notebook('notebooks/examples.ipynb')