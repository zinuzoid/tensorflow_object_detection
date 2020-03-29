from setuptools import setup, find_packages
setup(
    name="tensorflow_object_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=['opencv-python==3.4.6.27']
)

# pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
