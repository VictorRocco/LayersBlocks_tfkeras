install: all

all:
	echo "installing..."
	python3 setuptools_script.py bdist_wheel
	python3 -m pip install dist/*.whl
	python3 -m pip install LayersBlocks

clean:
	echo "cleaning..."
	pip3 uninstall -y LayersBlocks
	rm -rf LayersBlocks.egg-info/
	rm -rf build/
	rm -rf dist/

	
