#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n = 2000;
    int max_iter = 5000;
    double tol = 1e-8;
    unsigned seed = 12345;

    // Число потоков из аргументов или максимальное
    int threads = omp_get_max_threads();
    if (argc > 1) {
        threads = stoi(argv[1]);
        if (threads < 1) threads = 1;
        if (threads > omp_get_max_threads()) threads = omp_get_max_threads();
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    vector<double> Aflat((size_t)n * (size_t)n);
    auto A = [&](int i, int j)->double& {
        return Aflat[(size_t)i * n + j];
    };

    vector<double> b(n), x(n), xnew(n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ud(-1.0, 1.0);

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < n; ++j){
                A(i,j) = ud(rng);
            }
        }

        #pragma omp for schedule(static)
        for(int i = 0; i < n; ++i){
            double sum = 0.0;
            for(int j = 0; j < n; ++j){
                if (j != i) sum += fabs(A(i,j));
            }
            A(i,i) = sum + 1.0 + fabs(ud(rng));
            b[i] = ud(rng);
            x[i] = 0.0;
        }
    }

    auto tstart = chrono::high_resolution_clock::now();

    double normb = 0.0;
    for(int i=0; i<n; ++i) normb += b[i] * b[i];
    normb = sqrt(normb);
    if (normb == 0.0) normb = 1.0;

    int iter;
    double residual = 0;

    for(iter = 0; iter < max_iter; ++iter){

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; ++i){
            double s = 0.0;
            double diag = A(i,i);
            for(int j = 0; j < n; ++j){
                if (j != i)
                    s += A(i,j) * x[j];
            }
            xnew[i] = (b[i] - s) / diag;
        }

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; ++i)
            x[i] = xnew[i];

        double rnorm2 = 0.0;

        #pragma omp parallel for reduction(+:rnorm2) schedule(static)
        for(int i=0; i<n; ++i){
            double s = 0.0;
            for(int j=0; j<n; ++j)
                s += A(i,j) * x[j];
            double ri = s - b[i];
            rnorm2 += ri * ri;
        }

        residual = sqrt(rnorm2) / normb;
        if (residual < tol) break;
    }

    auto tfinish = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(tfinish - tstart).count();

    cout.setf(std::ios::fixed);
    cout << setprecision(6);

    cout << "n=" << n
         << " threads=" << threads
         << " iter=" << iter + 1
         << " residual=" << residual
         << " time=" << elapsed
         << "\n";

    {
        std::ofstream fout("results.csv", ios::app);
        fout << threads << "," << elapsed << "\n";
    }

    return 0;
}
