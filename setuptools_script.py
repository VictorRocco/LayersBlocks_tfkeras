import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LayersBlocks",
    version="0.3.32",
    description="Layers / Blocks for TensorFlow / Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VictorRocco/LayersBlocks_tfkeras",
    author="Victor Rocco",
    author_email="victor_rocco@hotmail.com",
    license="MIT",
    # Exclude the build files.
    packages=setuptools.find_packages(exclude=["tests"]),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
    zip_safe=True
)

