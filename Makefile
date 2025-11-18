.PHONY: all build test clean gtest

SRC = lab3.cpp
TEST = jacobi_solver_test.cpp
GTEST_DIR = googletest
GTEST_INC = $(GTEST_DIR)/googletest/include
GTEST_LIB = $(GTEST_DIR)/build/lib

all: build

# Компиляция основной программы
build:
	mkdir -p build
	g++ -std=c++17 -fopenmp $(SRC) -o build/app

# Скачивание и сборка googletest
gtest:
	if [ ! -d $(GTEST_DIR) ]; then git clone https://github.com/google/googletest $(GTEST_DIR); fi
	cd $(GTEST_DIR) && mkdir -p build && cd build && cmake .. && make -j4

# Компиляция и запуск тестов
test: gtest
	mkdir -p build
	g++ -std=c++17 -fopenmp \
	    -I$(GTEST_INC) \
	    $(TEST) $(SRC) \
	    -L$(GTEST_LIB) -lgtest -lgtest_main -lpthread \
	    -o build/tests.exe
	cd build && ./tests.exe --gtest_color=yes || true

# Очистка
clean:
	rm -rf build
