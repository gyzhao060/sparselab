============
Installation
============

Requirements
===============

Sparselab consists of python modules and Fortran/C internal libraries called from python modules. Here, we summarize required python packages and external packages for Sparselab.

You will also need **autoconf** and **`ds9`_** for compiling the library.

.. ds9: http://ds9.si.edu/site/Home.html

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
  **We strongly recommend using OpenBLAS**, which is the fastest library among publicly-available BLAS implementations. Our recommendation is to build up `OpenBLAS`_ by yourself with a compile option USE_OPENMP=1 and use it for our library. The option USE_OPENMP=1 enables OpenBLAS to perform paralleled multi-threads calculations, which will accelerate our library.

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

For compiling the whole library, you need to work in your Sparselab directory.

.. code-block:: Bash

  cd (Your Sparselab Directory)

A configure file can be generated with `autoconf`.

.. code-block:: Bash

  autoconf

Generate Makefiles with `./configure`. You might need `LDFLAGS` for links to BLAS and LAPACK.

.. code-block:: Bash

  # If you already have a library path to both BLAS and LAPACK.
  ./configure

  # If you don't have a PATH to BLAS and LAPACK, you can add links to them as follows
  ./configure LDFLAGS="-L(path-to-your-BLAS) -L(path-to-your-LAPACK)"

Make and compile the library. The internal C/Fortran Library will be compiled into python modules.

.. code-block:: Bash

  make install

Finally, please add a PYTHONPATH to your Sparselab Directory. We recommend to add a following line into your `.bashrc` (`.bash_profile` for Mac OS X) file.

.. code-block:: Bash

  # Add a python path to Sparselab
  export PYTHONPATH=$PYTHONPATH:(Your Sparselab Directory)

If you can load in your python interpretator, sparselab is probably installed successfully.

.. code-block:: Python

  # import sparselab
  from sparselab import imdata, uvdata, imaging


Updating Sparselab
==================

We recommend cleaning up the entire library before updating.

.. code-block:: Bash

  cd (Your Sparselab Directory)
  make uninstall

Then, you can update the repository with `git pull`.

.. code-block:: Bash

  git pull

Now, the repository has updated. You can follow :ref:`Installation` for recompiling your Sparselab.
