from setuptools import setup

setup(
    name='stratego_env',
    version='1.0',
    description='Stratego Env',
    author='J.B. Lanier, Stephen McAleer',
    author_email='jblanier@uci.edu',
    packages=['stratego_env'],
    install_requires=[
        'numpy',
        'eventlet',
        'dill',
        'gym',
        'python-socketio==4.6.0',  # https://github.com/JBLanier/stratego_env/issues/1
        'Flask-SocketIO==4.3.1',
        'python-engineio==3.13.2',
        'numba',
        'h5py',
        'requests'
    ],
)
