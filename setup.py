from setuptools import setup
from glob import glob
from setuptools import find_namespace_packages

setup(
    name="SegmentPy",
    version="0.1",
    description="Deep Learning Software for Tomography Segmentation",
    url="http://segmentpy.readthedocs.org",
    author="Zeliang Su, Arnaud Demortiere",
    author_email="zeliang.su@gmail.com, arnaud.demortiere@u-picardie.fr",
    license="Apache 2.0",
    classfiers=[
        'License :: OSI Approved :: Apache Software License v2.0',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_namespace_packages('src'),
    package_data={
        'img': glob('img/*.png'),
    },
    install_requires=[
        "numpy==1.19.1",
        "tensorflow-gpu==1.14",
        "pandas==1.1.1",
        "Pillow",
        # "openmpi",
        "mpi4py==3.0.3",
        "PySide2",
        "matplotlib==3.3.1",
        "scikit-learn==0.23.2",
        "scipy==1.5.2",
        "tqdm",
        "opencv-python==3.4.2.*",
        "h5py==2.8.0",
    ],
    keywords=['segmentation', 'CNN', 'XCT-image', 'battery'],
    python_requires="==3.6",
    entry_points={
        'console_scripts': [
            'segmentpy = SegmentPy:main'
        ]

    }
)

