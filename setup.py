import os
import re
from setuptools import setup


def get_version():
    with open(os.path.join('kapre', '__init__.py'), 'r', encoding='utf-8') as f:
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read())
        if match:
            return match.group(1)
        raise RuntimeError("Unable to find version string in kapre/__init__.py")


def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()


setup(
    name='kapre',
    version=get_version(),
    description='Kapre: Keras Audio Preprocessors. Tensorflow.Keras layers for audio pre-processing in deep learning',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Keunwoo Choi',
    url='https://github.com/keunwoochoi/kapre/',
    author_email='gnuchoi@gmail.com',
    license='MIT',
    packages=['kapre'],
    python_requires='>=3.9',
    install_requires=[
        'numpy >= 1.26.0, < 3.0.0',
        'librosa >= 0.11.0, < 1.0.0',
        # Keras 3.14.x currently breaks TensorFlow 2.20 TFLite conversion on Python 3.12+.
        'keras >= 3.0.0, < 3.14.0',
        'tensorflow >= 2.16.0, < 2.21.0',
    ],
    extras_require={
        'dev': [
            'black >= 24.0',
            'flake8 >= 7.0',
            'mypy >= 1.8',
            'pyright >= 1.1',
            'pytest >= 8.0',
        ],
        'docs': [
            'sphinx >= 7.0',
            'sphinx-rtd-theme >= 2.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='audio music speech sound deep learning keras tensorflow',
    zip_safe=False,
)
