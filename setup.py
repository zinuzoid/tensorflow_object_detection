from setuptools import setup, find_packages
setup(
    name="tensorflow_object_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=['opencv-python==3.4.9.31']
)
