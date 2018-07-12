"""A setuptools based setup module.
    
    See:
    https://packaging.python.org/en/latest/distributing.html
    https://github.com/pypa/sampleproject
    """

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os
from os import path
from setuptools.command.install import install
import subprocess

class MyInstall(install):
    def run(self):
        path = os.getcwd().replace(" ", "\ ").replace("(","\(").replace(")","\)") + "/bin/"
        subprocess.call(['echo', 'Compiling C++ code.'])
        subprocess.call(['chmod', '+x', os.path.join(path,'createGraphLibFile.sh')])
        #os.system("sh "+path+"createGraphLibFile.sh")
        subprocess.call(['sh',os.path.join(path,'createGraphLibFile.sh')])
        #os.system('cd localgraphclustering/graph_lib/lib/graph_lib_test; pwd; ls; make clean; make -f Makefile; cp libgraph.dylib ../../../../build/lib/localgraphclustering/graph_lib/lib/graph_lib_test/')
        install.run(self)

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
      name='localgraphclustering',
      
      # Versions should comply with PEP440.  For a discussion on single-sourcing
      # the version across setup.py and the project code, see
      # https://packaging.python.org/en/latest/single_source_version.html
      version='0.4.4',
      
      description='Package for local graph clustering',
      long_description=long_description,
      
      # The project's main homepage.
      url='https://github.com/kfoynt/LocalGraphClustering',
      
      # Author details
      author= 'Kimon Fountoulakis, Meng Liu, David Gleich, Michael W. Mahoney',
      author_email='kfount@berkeley.edu',
      
      # Choose your license
      license='GPL',
      
      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
                   # How mature is this project? Common values are
                   #   3 - Alpha
                   #   4 - Beta
                   #   5 - Production/Stable
                   'Development Status :: 3 - Alpha',
                   
                   # Pick your license as you wish (should match "license" above)
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   
                   # Specify the Python versions you support here. In particular, ensure
                   # that you indicate whether you support Python 2, Python 3 or both.
                   'Programming Language :: Python :: 3.5',
                   ],
      
      # What does your project relate to?
      keywords='local graph clustering',
      
      # You can just specify the packages manually here if your project is
      # simple. Or you can use find_packages().
      packages=find_packages(),

      # Alternatively, if you want to distribute just a my_module.py, uncomment
      # this:
      #   py_modules=["my_module"],
      
      # List run-time dependencies here.  These will be installed by pip when
      # your project is installed. For an analysis of "install_requires" vs pip's
      # requirements files see:
      # https://packaging.python.org/en/latest/requirements.html
      install_requires=[
                        'numpy >= 1.12.0',
                        'scipy >= 0.18.1',
                        #'csv >= 1.0',
                        'networkx >= 1.11',
                        #'time >= 0.7',
                        #'copy >= 2.5',
                        'matplotlib >= 2.0.0',
                        'typing',
                        'matplotlib',
                        'pandas',
                        'plotly'
                        ],
      
      # List additional groups of dependencies here (e.g. development
      # dependencies). You can install these using the following syntax,
      # for example:
      # $ pip install -e .[dev,test]
      #extras_require={
      #'dev': ['check-manifest'],
      #'test': ['coverage'],
      #},
      
      include_package_data=True,
      
      # If there are data files included in your packages that need to be
      # installed, specify them here.  If using Python 2.6 or less, then these
      # have to be included in MANIFEST.in as well.
      package_data={
      'localgraphclustering.src': ['*'],
      'localgraphclustering.src.lib.graph_lib_test': ['*'],
      'localgraphclustering.src.lib.graph_lib_test': ['*.dylib'],
      'localgraphclustering.src.lib.graph_lib_test': ['Makefile'],
      'localgraphclustering.interface': ['*.py'],
      'localgraphclustering.interface.types': ['*.py']
      },
      
      # Although 'package_data' is the preferred approach, in some case you may
      # need to place data files outside of your packages. See:
      # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
      # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
      # data_files=[('my_data', ['data/data_file'])],
      
      # To provide executable scripts, use entry points in preference to the
      # "scripts" keyword. Entry points provide cross-platform support and allow
      # pip to create the appropriate form of executable for the target platform.
      #entry_points={
      #'console_scripts': [
      #                    'localgraphclustering=sample:main',
      #                    ],
      #},
      
      scripts=['bin/createGraphLibFile.sh'],
      cmdclass={'install': MyInstall},
      )
