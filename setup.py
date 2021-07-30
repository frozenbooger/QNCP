from setuptools import setup, find_packages


setup(
    name='QNCP',
    version='1.0',
    license='MIT',
    author="Leonardo Castillo",
    author_email='frozenbooger@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/frozenbooger/QNCP',
    keywords='Quantum Network Control Panel',
    install_requires=[
        'matplotlib',
        'numpy',
        'numpydoc',
        'paramiko',
        'pyvis',
        'PyVISA',
        'PyVISA-py',
        'regex',
        'scipy',
        'scp',
        'visa'],

)