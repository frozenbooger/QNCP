from setuptools import setup, find_packages


setup(
    name='QNCP',
    version='1.9.7',#first: when we get to 10 on second, second: new devices, new package-req., new-functions, three: bugs, four: minor bugs
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
        'scp',
        'visa',
        'python-vxi11'],

)