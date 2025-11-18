.PHONY: all build test clean

SRC = lab3.cpp
TEST = jacobi_tests.cpp
GTEST_DIR = googletest

all: build

build:
	mkdir -p build
	g++ -std=c++17 -fopenmp $(SRC) -o build/app

test:
	mkdir -p build
	# скачиваем googletest, если ещё нет
	if [ ! -d $(GTEST_DIR) ]; then git clone https://github.com/google/googletest $(GTEST_DIR); fi
	cd $(GTEST_DIR) && mkdir -p build && cd build && cmake .. && make -j4
	g++ -std=c++17 -fopenmp -I./$(GTEST_DIR)/googletest/include \
		$(TEST) $(SRC) \
		-L./$(GTEST_DIR)/build/lib -lgtest -lgtest_main -lpthread \
		-o build/tests.exe
	cd build && ./tests.exe

clean:
	rm -rf build
