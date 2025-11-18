.PHONY: all build test clean

GTEST_INC = googletest/googletest/include
GTEST_LIB = googletest/build/lib

all: build

build:
	mkdir -p build
	g++ -std=c++17 -fopenmp lab3.cpp -o build/app

test:
	mkdir -p build
	g++ -std=c++17 -fopenmp \
		-I$(GTEST_INC) \
		jacobi_solver_test.cpp lab3.cpp \
		-L$(GTEST_LIB) -lgtest -lgtest_main -lpthread \
		-o build/tests.exe
	cd build && ./tests.exe --gtest_color=yes

clean:
	rm -rf build
