from setuptools import setup

setup(
    name="SegmentPy",
    version="0.1",
    description="Deep Learning Software for Tomography Segmentation",
    url="https://github.com/ZeliangSu",
    author="Zeliang Su",
    author_email="zeliang.su@u-picardie.fr",
    license="MIT",
    packages=["tf114"],
    install_requires=[
        "tensorflow==1.14",
        "pandas",
        "Pillow",
        "openmpi",
        "mpi4py",
        "PyQt5",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "tqdm",
        "pyinstaller"
    ],
    python_requires=">=3.6",
    zip_safe=False,
    entry_points={
        'what': [
            'SegmentPy = package.module:SegmentPy.py'
        ]
    }
)

