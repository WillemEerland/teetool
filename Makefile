all: clean install clean test doc

clean:
	rm -rf build
	rm -f */*.pyc

install:
	pip install -e .

test: test/*.py teetool/*.py
	py.test -v -s

doc: test/*.py teetool/*.py
	doxygen Doxyfile
