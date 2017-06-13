============
Installation
============

Requirements
===============

Sparselab consists of python modules and Fortran/C internal libraries called from python modules. Here, we summarize required python packages and external packages for Sparselab.

You will also need **autoconf** for compiling the library.

Python Packages and Modules
---------------------------

Sparselab has been tested in Python 2.7. Sparselab uses **numpy**, **scipy**, **matplotlib**, **pandas**, **astropy**, **xarrays** and **pyds9**. Sparselab has been tested and developped in Python 2.7 environments provided by the `Anaconda`_ package that includes required packages except *xarrays** and **pyds9*. We recommend using Anaconda for Sparselab.

.. _Anaconda: https://www.continuum.io/anaconda-overview

**xarrays** and **pyds9** can be installed with pip as follows (see the official websites of `xarray`_ and `pyds9`_ for installation).

.. code-block:: Bash

  pip install xarrays
  pip install git+https://github.com/ericmandel/pyds9.git#egg=pyds9

.. _xarray: http://xarray.pydata.org/en/stable/
.. _pyds9: https://github.com/ericmandel/pyds9


External Libraries
------------------

Fortran/C internal libraries of Sparselab use following external libraries.

1) BLAS
  **We strongly recommend using OpenBLAS, which is the fastest library among publicly available BLAS implementations**. Our recommendation is to build up `OpenBLAS`_ by yourself with a compile option USE_OPENMP=1 and use it for our library. The option USE_OPENMP=1 enables OpenBLAS to perform paralleled multi-threads calculations, which will accelerate our library.

.. _OpenBLAS: https://github.com/xianyi/OpenBLAS

2) LAPACK
  LAPACK does not have a big impact on computational costs of imaging. The default LAPACK package in your Linux/OS X package system would be acceptable for Spareselab. Of course, you may build up `LAPACK`_ by yourself.

.. _LAPACK: https://github.com/Reference-LAPACK/lapack-release


Download, Install and Update
============================

Downloading Sparselab
---------------------
You can download the code from github.

.. code-block:: Bash

  # Clone the repository
  git clone https://github.com/eht-jp/sparselab

.. _Installation:
Installation
------------

0) Go to your Sparselab directory.

.. code-block:: Bash

  cd (Your Sparselab Directory)

1) Genarate a configure file with autoconf.

.. code-block:: Bash

  autoconf

2) Configure make files with `./configure`. You might need `LDFLAGS` for links to BLAS and LAPACK.

.. code-block:: Bash

  # If you already have a library path to both BLAS and LAPACK.
  ./configure

  # If you don't have a PATH to BLAS and LAPACK, you can add links to them as follows
  ./configure LDFLAGS="-L(path-to-your-BLAS) -L(path-to-your-LAPACK)"

3) Compile the library. The internal C/Fortran Library will be compiled into python modules.

.. code-block:: Bash

  make install

4) Finally, add a PYTHONPATH to your Sparselab Directory. We recommend to add a following line into your `.bashrc` (`.bash_profile` for Mac OS X) file.

.. code-block:: Bash

  # Add a python path to Sparselab
  export PYTHONPATH=$PYTHONPATH:(Your Sparselab Directory)


Updating Sparselab
==================

1) Clean up the entire library before updating.

.. code-block:: Bash

  cd (Your Sparselab Directory)
  make uninstall

2) Update the repository with `git pull`.

.. code-block:: Bash

  git pull

3) You can follow :ref:`Installation` for recompiling your Sparselab.