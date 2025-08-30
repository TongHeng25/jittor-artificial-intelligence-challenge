from setuptools import setup, find_packages

setup(
    name='bone_inspector',
    version='0.1.0',
    description='simple package to transfer 3d motion',
    author='czpcf',
    package_dir={'': 'src'},
    packages=find_packages(),
    install_requires=[
        'numpy',
        'trimesh',
        'scipy',
        'bpy==4.2',
        'fast_simplification',
    ],
    python_requires='>=3.11',
)
