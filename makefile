module = LayersBlocks
directories = LayersBlocks

pre_commit:
	echo "pre commit..."
	isort $(directories)
	black $(directories)
	##mypy $(directories) --ignore-missing-imports --strict
	#flake8 $(directories) --max-line-length 110 --max-complexity 10 --extend-ignore=F405,F403,E203
	flake8 $(directories) --max-line-length 110 --max-complexity 10 --extend-ignore=C901
	pylint $(directories) --disable=E0401

all: clean install

install:
	echo "installing..."
	python3 setuptools_script.py bdist_wheel
	python3 -m pip install dist/*.whl
	python3 -m pip install $(module)

clean:
	echo "cleaning..."
	pip3 uninstall -y $(module)
	rm -rf $(module).egg-info/
	rm -rf build/
	rm -rf dist/



