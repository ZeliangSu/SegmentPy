from setuptools import setup

setup(
    name="SegmentPy",
    version="0.1",
    description="Deep Learning Software for Tomography Segmentation",
    url="https://github.com/ZeliangSu",
    author="Zeliang Su, Arnaud Demortiere",
    author_email="zeliang.su@gmail.com, arnaud.demortiere@u-picardie.fr",
    license="Apache 2.0",
    install_requires=[
        "tensorflow==1.14",
        "pandas",
        "Pillow",
        # "openmpi",
        "mpi4py",
        "PySide2",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "tqdm",
        "opencv",
        # "pyinstaller",
    ],
    python_requires=">=3.6",
    # zip_safe=False,
    entry_points={
        'console_scripts': [
            'SegmentPy = src.SegmentPy:main'
        ]

    }
)

