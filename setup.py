from setuptools import setup, find_packages

setup(
    name='translation',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'torch==1.13.1',
        'datasets',
        'torchvision',
        'torchaudio',
        'transformers',
        'pytorch-lightning',
        'scikit-learn',
        'matplotlib',
        'jupyter',
        'pandas',
        'sentencepiece',
        'sacremoses'
    ],
    url='',
    license='',
    author='William Sutcliffe',
    author_email='william.sutcliffe08@gmail.com',
    description='Package utilizing transformers for various NLP tasks including language translation and sentiment analysis.'
)
