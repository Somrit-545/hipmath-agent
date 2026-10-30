import os
import sysconfig
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

HIP_INCLUDE  = "/opt/rocm-6.2.0/include"
CUDA_INCLUDE = "/usr/local/cuda/include"

class HipBuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            self.build_hip_extension(ext)

    def build_hip_extension(self, ext):
        os.makedirs(self.build_temp, exist_ok=True)
        objs = []
        for src in ext.sources:
            base = os.path.basename(src)
            if src.endswith(".cu"):
                obj = os.path.join(self.build_temp, base.replace(".cu", ".o"))
            else:
                obj = os.path.join(self.build_temp, base.replace(".cpp", ".o"))
                
            # USE HIPCC FOR BOTH .cu AND .cpp
            cmd = ["hipcc", "-c", src, "-o", obj, "-fPIC", "-O3", "-std=c++17",
                   "-D__HIP_PLATFORM_NVIDIA__",
                   f"-I{HIP_INCLUDE}", f"-I{CUDA_INCLUDE}",
                   f"-I{pybind11.get_include()}",
                   f"-I{os.path.dirname(src)}",
                   f"-I{sysconfig.get_paths()['include']}"]
                   
            subprocess.check_call(cmd)
            objs.append(obj)

        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        
        # USE HIPCC FOR LINKING
        subprocess.check_call(
            ["hipcc", "-shared", "-o", ext_path] + objs +
            ["-L/usr/local/cuda/lib64", "-lcudart",
             "-L/opt/rocm-6.2.0/lib", "-lamdhip64"]
        )

setup(
    name="hipmath",
    version="0.1.0",
    ext_modules=[
        Extension(
            "_hipmath",
            sources=["../src/hip_kernels.cu", "../src/bindings.cpp"],
            include_dirs=["../src"],
        )
    ],
    cmdclass={"build_ext": HipBuildExt},
    packages=["hipmath"],
    package_dir={"hipmath": "hipmath"},
    entry_points={"console_scripts": ["hipmath=hipmath.cli:main"]},
    zip_safe=False,
)
