all: clean install clean test doc

clean:
	rm -rf build
	rm -f */*.pyc

install:
	python setup.py install

test:
	py.test -v -s

doc:
	doxygen Doxyfile
