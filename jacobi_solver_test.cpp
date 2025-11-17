#include <gtest/gtest.h>
#include <sstream> // чтобы перенаправить вывод std::cout в строку
#include <string>
#include <iostream>
#include <vector>
#include <omp.h>

// Подключаем main.cpp, но с помощью макроса переименовываем его в jacobi_original_main,
// чтобы он не конфликтовал с main в текущем файле с Google Test, 
// т.е. перед компиляцией все слова main будут заменены на jacobi_original_main
#define main jacobi_original_main
#include "main.cpp"
#undef main // отменяем этот макрос

struct JacobiRunResult {
  int ret_code{}; // return 0, другое значение - ошибка
  int n{}; 
  int threads{}; // кол-во потоков OpenMP, кот-ые использовались программлй
  int iter{};
  double residual{}; // относительная невязка
  double time{}; // время работы основного алгоритма (итерации метода Якоби + вычисление невязки), в секундах
};

// Вспомогательная функция, кот-ая отвечает за то,
// что мы можем вытащить из результата нужное нам значение по ключу (для int)
// на входе ф-ия получает сам результат и тот ключ, значение кот-го нас интересует
// например, ParseIntField(line, "iter=") и получим 100 (например)
// static - ф-ия видна только внутри текущего файла
static int ParseIntField(const std::string& line, const std::string& key) {
  auto pos = line.find(key);
  if (pos == std::string::npos) {
    throw std::runtime_error("Key not found: " + key);
  }
  pos += key.size();
  auto end = line.find_first_of(" \t\r\n", pos);
  std::string value = line.substr(pos, end - pos);
  return std::stoi(value);
}

// то же, что и выше, но для double
static double ParseDoubleField(const std::string& line, const std::string& key) {
  auto pos = line.find(key);
  if (pos == std::string::npos) {
    throw std::runtime_error("Key not found: " + key);
  }
  pos += key.size();
  auto end = line.find_first_of(" \t\r\n", pos);
  std::string value = line.substr(pos, end - pos);
  return std::stod(value);
}

// Симуляция запуска программы с командной строки с заданными аргументами
// она сбирает массив из этих аргументов, будто бы мы ввели его в командную строку
// tol - точность, seed - значение для генератора
static JacobiRunResult RunJacobi(int n, int max_iter, double tol, unsigned seed) {
  std::vector<std::string> args;
  args.emplace_back("prog"); // не важно, по соглашению argv[0] всегда чем-то заполнен
  args.emplace_back(std::to_string(n)); // argv[1]
  args.emplace_back(std::to_string(max_iter)); // argv[2]
  args.emplace_back(std::to_string(tol)); // argv[3]
  args.emplace_back(std::to_string(seed)); // argv[4]


  std::vector<char*> argv; // вектор указателей на строки (точнее на первый символ каждой строк)
  argv.reserve(args.size());
  for (auto& s : args) {
    argv.push_back(s.data()); // char**, указатель на первый элемент
  }

  // перенаправляем cout в строковый поток
  std::ostringstream oss; // класс из <sstream>, поток, который пишет в строку, 
  // т.е. вместо выода в консоль, вывод копится внутри oss в памяти
  // мы записываем все в буфер:
  std::streambuf* old_buf = std::cout.rdbuf(oss.rdbuf()); // сохраняем старый буфер в old_buf, это буфер консоли
  // при вызове main, cout теперь «подключен» к oss и сё складывается в строковый буфер

  int ret = jacobi_original_main(static_cast<int>(argv.size()), argv.data());

  std::cout.rdbuf(old_buf); // возвращаемся к буферу консоли

  std::string out = oss.str(); // сохраниям в переменную данный из строквого буфера

  JacobiRunResult res;
  res.ret_code = ret;
  res.n = ParseIntField(out, "n=");
  res.threads = ParseIntField(out, "threads=");
  res.iter = ParseIntField(out, "iter=");
  res.residual = ParseDoubleField(out, "residual=");
  res.time = ParseDoubleField(out, "time=");

  return res;
}

// 1) Проверяем, отрабатывает ли программа случай, когда пользователь вводит свое значение n
// Параметр n в выводе должен совпадать с переданным аргументом
TEST(JacobiProgramTest, NArgumentIsRespected) {
  int n = 20;
  auto res = RunJacobi(n, /*max_iter=*/5000, /*tol=*/1e-8, /*seed=*/12345);
  EXPECT_EQ(res.ret_code, 0);
  EXPECT_EQ(res.n, n);
}

// 2) Проверяем, что программа корректно определяет количество потоков через omp_get_max_threads() и сохраняет его в переменную threads
TEST(JacobiProgramTest, ThreadsMatchOmpMaxThreads) {
  auto res = RunJacobi(10, /*max_iter=*/100, /*tol=*/1e-6, /*seed=*/123u);
  int expected_threads = omp_get_max_threads();
  EXPECT_EQ(res.threads, expected_threads);
}

// 3)) Проверяем, что количество итераций не больше заданного max_iter
TEST(JacobiProgramTest, IterNotGreaterThanMaxIter) {
  int max_iter = 15;
  auto res = RunJacobi(10, max_iter, /*tol=*/1e-6, /*seed=*/1u);
  EXPECT_LE(res.iter, max_iter);
}

// 4) Проверяем, что residual неотрицательный
TEST(JacobiProgramTest, ResidualIsNonNegative) {
  auto res = RunJacobi(10, /*max_iter=*/100, /*tol=*/1e-6, /*seed=*/42u);
  EXPECT_GE(res.residual, 0.0);
}

// 5) Проверяем, что если мы хотим строгую точность (меньший tol), 
// то шагов итераций должно быть >=, чем при грубой точности (больший tol)
TEST(JacobiProgramTest, IterationsDependOnTolerance) {
  int n = 20;
  int max_iter = 500;

  auto res_rude_tol = RunJacobi(n, max_iter, /*tol=*/1e-1, /*seed=*/777u); // грубая
  auto res_strict_tol = RunJacobi(n, max_iter, /*tol=*/1e-10, /*seed=*/777u); // строгая

  // При строгой точности итераций >=
  EXPECT_LE(res_rude_tol.iter, res_strict_tol.iter);
}








