from setuptools import find_packages, setup

packages = find_packages()
print("packages: ", packages)
setup(
    name="tecovllm_models",
    version="0.0.0",
    ext_modules=[],
    install_requires=[],
    packages=packages,
    entry_points={
        "vllm.general_plugins": ["sdaa = tecovllm_models:init"],
    },
)

