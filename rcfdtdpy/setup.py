from setuptools import setup

setup(name='rcfdtdpy',
      version='0.2.6',
      url='https://rcfdtdpy.readthedocs.io/en/latest/index.html',
      project_urls={
            'Documentation': 'https://rcfdtdpy.readthedocs.io/en/latest/index.html',
            'Source': 'https://github.com/jroth137/rcfdtdpy',
      },
      author='Jack Roth',
      author_email='jroth@slac.stanford.edu',
      packages=['rcfdtdpy'],
      python_requires='>=3',
      install_requires=['numpy>=1.15.0', 'scipy>=1.1.0', 'tqdm>=4.24.0'],
      zip_safe=False)