#nsml: nvcr.io/nvidia/pytorch:20.03-py3
from distutils.core import setup

setup(
    name='push_test_torch',
    version='1.0',
    install_requires=[
        'torch==1.9.1',
        'torchvision==0.10.0',
        'tqdm',
        'pandas',
        'pillow',
    ]
)
