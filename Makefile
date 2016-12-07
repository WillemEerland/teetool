clean:
	rm -r build
	rm */*.pyc

install:
	python setup.py install

test:
	py.test -v -s
