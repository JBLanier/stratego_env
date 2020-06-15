from setuptools import setup

setup(
    name='stratego_gym',
    version='1.0',
    description='Stratego Gym',
    author='J.B. Lanier, Stephen McAleer',
    author_email='jblanier@uci.edu',
    packages=['stratego_gym'],
    install_requires=['numpy', 'eventlet', 'dill', 'gym', 'python-socketio', 'numba', 'h5py', 'requests'],
)
