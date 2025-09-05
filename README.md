# LatPy

This python module, named LatPy, is a python module that provides many lattice algorithms for experiment, implementation, or analytics. For example, lattice reductions, solving lattice problems.

## How to Install

First, you have to install two C++ libraries, that is, NTL library and Eingen library because this module makes use of NTL library and Eingen library in c++ sources.
You can easily install these libraries with ``apt`` command.

```bash
# installs Eigen library
$ sudo apt install libeigen3-dev

# installs NTL library
$ sudo apt-get install -y libntl-dev
```

If you finished installing these libraries, you are ready to install LatPy. You can install this python module with ``pip`` command. It seems that there are two ways to install this python module.

## Directly Install from Git Repository (Recommended)

You can directly install LatPy with url of this git repository using the below command.

```bash
$ pip install git+https://github.com/LatPy/LatPy.git
```

## Clone And Install (Non-Recommended)

You can install LatPy with cloning to local environment.

```bash
$ git clone https://github.com/LatPy/LatPy.git

$ cd LatPy

LatPy$ pip install .
```