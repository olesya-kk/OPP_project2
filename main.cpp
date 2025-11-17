// Подключаем стандартные заголовки C++ и OpenMP
#include <bits/stdc++.h>      // содержит большинство стандартных библиотек
#include <omp.h>              // подключение OpenMP для параллелизации

using namespace std;

// Решаем систему Ax = b методом Якоби (простых итераций)
// Формула: x_{k+1} = D^{-1}(b - (A - D)x_k)
// Матрица A генерируется случайно, но делается диагонально доминируемой для сходимости
int main(int argc, char** argv) {
    int n = 2000;                  // размер системы по умолчанию
    int max_iter = 5000;           // максимальное число итераций
    double tol = 1e-8;             // требуемая относительная точность
    unsigned seed = 12345;         // seed для генератора случайных чисел

    // Если пользователь передал аргументы — используем их
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) max_iter = atoi(argv[2]);
    if (argc > 3) tol = atof(argv[3]);
    if (argc > 4) seed = (unsigned)atoi(argv[4]);

    omp_set_dynamic(0);                    // Запрещает OpenMP менять число нитей динамически
    int threads = omp_get_max_threads();   // Получаем число потоков, доступных программе

    // Создаём плоский массив для матрицы A (размер n×n)
    // Храним как одномерный массив для лучшей кеш-локальности
    vector<double> Aflat((size_t)n * (size_t)n);

    // Лямбда-функция для удобного доступа A(i,j)
    auto A = [&](int i, int j) -> double& {
        return Aflat[(size_t)i * n + j];
    };

    // Векторы правой части b, текущего x и следующего xnew
    vector<double> b(n), x(n), xnew(n);

    // Инициализируем распределение случайных чисел
    std::uniform_real_distribution<double> ud(-1.0, 1.0);

    // ---------- ПАРАЛЛЕЛЬНОЕ СОЗДАНИЕ МАТРИЦЫ И ВЕКТОРА ----------
    #pragma omp parallel
    {
        // У каждого потока свой генератор, инициализированный от общего seed
        int tid = omp_get_thread_num();
        std::mt19937 rng(seed + tid);

        // Распараллеливаем заполнение матрицы A случайными числами
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A(i,j) = ud(rng);  // случайное число от -1 до 1
            }
        }

        // Корректируем диагональ, чтобы матрица стала диагонально доминируемой
        // Генерируем вектор b, инициализируем x = 0
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            // Суммируем абсолютные значения всех недиагональных элементов строки
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum += fabs(A(i,j));
                }
            }
            // Делаем диагональный элемент строго больше суммы остальных
            A(i,i) = sum + 1.0 + fabs(ud(rng));

            b[i] = ud(rng);    // свободный член
            x[i] = 0.0;        // начальное приближение
        }
    } // конец параллельной инициализации

    // Засекаем старт времени
    auto tstart = chrono::high_resolution_clock::now();

    // Вычисляем норму b (нужно для относительной невязки)
    double normb = 0.0;
    for (int i = 0; i < n; ++i) {
        normb += b[i] * b[i];
    }
    normb = sqrt(normb);
    if (normb == 0.0) {
        normb = 1.0; // избегаем деления на 0
    }

    int iter;            // фактическое число итераций
    double residual = 0; // относительная невязка

    // ------------------ ОСНОВНОЙ ИТЕРАЦИОННЫЙ ЦИКЛ ------------------
    for (iter = 0; iter < max_iter; ++iter) {

        // --------- ШАГ 1: вычисление нового вектора xnew (параллельно) ---------
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double s = 0.0;         // сумма A(i,j)*x[j]
            double diag = A(i,i);   // диагональный элемент A(i,i)

            // Суммируем вклад всех j ≠ i
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    s += A(i,j) * x[j];
                }
            }

            // Формула Якоби: xnew[i] = (b[i] - сумма) / A(ii)
            xnew[i] = (b[i] - s) / diag;
        }

        // --------- ШАГ 2: копируем xnew → x (параллельно) ---------
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            x[i] = xnew[i];
        }

        // --------- ШАГ 3: считаем невязку r = Ax - b (параллельно + reduction) ---------
        double rnorm2 = 0.0;

        #pragma omp parallel for reduction(+:rnorm2) schedule(static)
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            // Считаем произведение строки A на вектор x
            for (int j = 0; j < n; ++j) {
                s += A(i,j) * x[j];
            }

            double ri = s - b[i];   // компонента невязки r[i]
            rnorm2 += ri * ri;      // суммируем квадрат
        }

        // относительная невязка
        residual = sqrt(rnorm2) / normb;

        // проверка сходимости — если невязка достаточно мала, выходим
        if (residual < tol) {
            break;
        }
    }

    // Засекаем конец времени
    auto tfinish = chrono::high_resolution_clock::now();

    // Вычисляем время выполнения
    double elapsed = chrono::duration<double>(tfinish - tstart).count();

    // Форматируем вывод
    cout.setf(std::ios::fixed);
    cout << setprecision(6);

    // Выводим параметры, число потоков, число итераций и время
    cout << "n=" << n
         << " threads=" << threads
         << " iter=" << iter + 1
         << " residual=" << residual
         << " time=" << elapsed
         << "\n";

    return 0;   // завершение программы
}
