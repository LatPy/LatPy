from setuptools import setup, Extension

latpy = [
    Extension(
        "latpy.core._core",
        sources=[
            "latpy/core/src/globals.cpp",
            "latpy/core/src/compute_gso.cpp",
            "latpy/core/src/volume.cpp",
            "latpy/core/src/sl.cpp",
            "latpy/core/src/pot.cpp",
            "latpy/core/src/hf.cpp",
            "latpy/core/src/rhf.cpp"
        ],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-fopenmp",
            "-march=native",
            "-funroll-loops",
            "-lntl"
        ],
        extra_link_args=[
            "-fopenmp",
            "-lntl"
        ],
        include_dirs=[
            "latpy/core/include"
        ]
    ),
    Extension(
        "latpy._latpy",
        sources=[
            "src/latpy.cpp"
        ],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-fopenmp"
        ],
        extra_link_args=["-fopenmp"],
        include_dirs=[
            "include",
        ]
    )
]

setup(
    name="latpy",
    version="0.1.0",
    description="Example ctypes-based library",
    packages=["latpy"],
    ext_modules=latpy,
    python_requires=">=3.8",
)
