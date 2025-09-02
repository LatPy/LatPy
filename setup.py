from setuptools import setup, Extension

latpy = [
    Extension(
        "latpy.core._core",
        sources=[
            "latpy/core/src/compute_gso.cpp"
        ],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-fopenmp"
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
