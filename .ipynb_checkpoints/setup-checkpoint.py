from setuptools import setup, find_packages
import os, sys, pathlib

setup(
    name='QNCP',
    version='1.11',
    license='MIT',
    author="Leonardo Castillo Veneros and Guodong Cui",
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
        'scikit-learn',
        'scp',
        'visa',
        'python-vxi11'],
    extras_require = {
        'crystal fitting':  ["torch>=1.8"]},
    package_data={os.path.join('src','qncp'):['*.dll','qncp/dependencies/*']},
    data_files = [('dlls',[os.path.join('src','qncp','dependencies','tdcbase_64bit.dll'),os.path.join('src','qncp','dependencies','FTD3XX.dll'),os.path.join('src','qncp','dependencies','libusb0.dll')])],
    zip_safe=False
)