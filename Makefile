.PHONY: all build test clean

all: build

build:
	mkdir -p build
	g++ -std=c++17 -fopenmp main.cpp -o build/app

test:
	mkdir -p build
	g++ -std=c++17 -fopenmp -I./googletest/googletest/include \
		jacobi_solver_test.cpp lab3.cpp \
		-L./googletest/build/lib -lgtest -lgtest_main -lpthread \
		-o build/tests.exe
	cd build && ./tests.exe --gtest_color=yes

clean:
	rm -rf build
