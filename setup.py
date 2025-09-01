from setuptools import setup, Extension

mylib = Extension(
    "latpy._latpy",     # パッケージ内に配置
    sources=["src/latpy.cpp"],
    language="c++",
    extra_compile_args=["-O3"],
    include_dirs=["include"]
)

setup(
    name="latpy",
    version="0.1.0",
    description="Example ctypes-based library",
    packages=["latpy"],
    ext_modules=[mylib],
    python_requires=">=3.8",
)
