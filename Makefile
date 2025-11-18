.PHONY: all build test clean gtest

GTEST_DIR = googletest
GTEST_INC = $(GTEST_DIR)/googletest/include
GTEST_LIB = $(GTEST_DIR)/build/lib

all: build

build:
	mkdir -p build
	g++ -std=c++17 -fopenmp lab3.cpp -o build/app

# -----------------------------
# Сборка googletest
# -----------------------------
gtest:
	if [ ! -d $(GTEST_DIR) ]; then git clone https://github.com/google/googletest $(GTEST_DIR); fi
	cd $(GTEST_DIR) && mkdir -p build && cd build && cmake .. && make -j4

# -----------------------------
# Компиляция и запуск тестов
# -----------------------------
test: gtest
	mkdir -p build
	g++ -std=c++17 -fopenmp \
		-I$(GTEST_INC) \
		jacobi_solver_test.cpp lab3.cpp \
		-L$(GTEST_LIB) -lgtest -lgtest_main -lpthread \
		-o build/tests.exe
	cd build && ./tests.exe --gtest_color=yes

clean:
	rm -rf build
