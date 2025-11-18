.PHONY: all build test clean

SRC = lab3.cpp
TEST = jacobi_solver_test.cpp
GTEST_DIR = googletest

all: build

build:
	mkdir -p build
	g++ -std=c++17 -fopenmp -I./include $(SRC) -o build/app

test:
	mkdir -p build
	g++ -std=c++17 -fopenmp -I./$(GTEST_DIR)/googletest/include \
	    $(TEST) lab3.cpp \
	    -L./$(GTEST_DIR)/build/lib -lgtest -lgtest_main -lpthread \
	    -o build/tests.exe
	cd build && ./tests.exe

clean:
	rm -rf build
