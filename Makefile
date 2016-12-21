all: clean install clean test doc

clean:
	rm -rf build
	rm -f */*.pyc

install:
	python setup.py install

test: test/*.py teetool/*.py
	py.test -v -s
	py.test --nbval example/example_toy_2d.ipynb

doc:
	doxygen Doxyfile
