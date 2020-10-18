import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="neural-style-transfer",
    version="0.0.1",
    description="Reimplementaion of Neural Style Transfer by tesorflow.org",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harsh020/neural-style-transfer.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'tensorflow',
          'numpy',
          'regex',
          'pillow'
      ],
    python_requires='>=3.6',
)
