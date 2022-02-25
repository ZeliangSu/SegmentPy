=====================================
Installation & Setup
=====================================
We highly recommand to use GPUs for the training. CUDA and MPI should be pre-installed in the device.

.. .. toctree::
..    :maxdepth: 3
..    :caption: Installation & Setup

..    Pre-requirements
..    Linux or MacOS
..    Windows
..    Frequent installation question

Pre-requirements
---------------------
1. nvidia gpu driver
2. cudatoolkit 10.0 (for the current implemented tensorflow version)

.. image:: img/tensorflow_cuda_compatible.png
   :width: 400

3. openmpi 

Linux or MacOS
--------------
Installation by command 

.. warning::
   The interface is an *testing alpha* version, we invite you for now to install from source. 
   See below for installation from source.


Installation from source:

1. go to https://github.com/ZeliangSu/SegmentPy and copy the HTTPS link

.. image:: img/download_from_git.png
  :width: 400
  
and download by typing::

   $ git clone https://github.com/ZeliangSu/SegmentPy.git
   $ cd to/the/path/of/the/git/cloned/SegmentPy
   $ python setup.py install

2. or download the zip file and zip it, then::

   $ cd to/the/path/of/the/unzipped/SegmentPy
   $ python setup.py install

Launch the SegmentPy by typing in the Terminal::

   $ segmentpy

Frequent installation questions
---------------------------------

Q: Can SegmentPy run on a Windows workstation?

A: Unfortunately, we could not at the current state garentee the performance on Windows machine. We recently gathered some expected behavior reports on Windows and are currently working on it. 