# LatPy

This is now developping yet. So this has some defects and algorithms that is not implemented.

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

You can directly install LatPy with url of this git repository using the below command. (Because install is expected to take a few long time, the below command includs ``-v`` option for never getting tired of waiting.)

```bash
$ pip install git+https://github.com/LatPy/LatPy.git -v
```

## Clone And Install (Non-Recommended)

You can install LatPy with cloning to local environment.

```bash
# clones LatPy to local environment
$ git clone https://github.com/LatPy/LatPy.git

# changes directories to LatPy directories
$ cd LatPy

# installs LatPy
LatPy$ pip install .
```

## Check If Correctly Installed

If you want to check if LatPy was correctly installed, you can check it with running a simple test code, for example the below code.

```python
import latpy
import numpy as np

B = np.random.randint(100, 999, size=(50, 50), dtype=np.int64)

C = latpy.Lattice(B)
print(C)
print(C.compute_gso())
print(C.sl())
print(C.pot())
print(C.size(0.5))
print(C.lll())
```

If the above code behaves with no problem, it seems that LatPy was correctly installed.

## Expected Trouble Shooting

If an eeror like the below occured:

```bash
error: subprocess-exited-with-error 
× python setup.py bdist_wheel did not run successfully. 
│ exit code: 1 
╰─> See above for output. 

note: This error originates from a subprocess, and is likely not a problem with pip. 
 Building wheel for latpy (setup.py) ... error 
 ERROR: Failed building wheel for latpy 
 Running setup.py clean for latpy 
Failed to build latpy 
ERROR: ERROR: Failed to build installable wheels for some pyproject.toml based projects (latpy)
```

then it seems that you did not install c++ libraries. Please install NTL library and Eigen library.

## Disclaimer

The all developers of LatPy are NOT responsible for any losses, damages, deficits, troubles or disadvantages that was caused by installing, cloning or using this python module, that is, LatPy.
