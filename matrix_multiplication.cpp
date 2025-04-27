#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <iomanip>  // For setprecision
#include <cmath>    // For isfinite

using namespace std;
using namespace std::chrono;

void initializeMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % 100;
        }
    }
}

void printMatrix(const vector<vector<int>>& matrix, const string& name) {
    if (matrix.empty() || matrix[0].empty()) return;
    
    cout << "\nMatrix " << name << " (" << matrix.size() 
         << "x" << matrix[0].size() << "):\n";
    for (const auto& row : matrix) {
        for (int val : row) {
            cout << setw(4) << val << " ";
        }
        cout << "\n";
    }
}

void sequentialMultiply(const vector<vector<int>>& A, 
                       const vector<vector<int>>& B,
                       vector<vector<int>>& C, 
                       int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallelMultiply(const vector<vector<int>>& A, 
                     const vector<vector<int>>& B,
                     vector<vector<int>>& C, 
                     int m, int n, int p) {
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    // Matrix dimensions
    int m, n, p;
    cout << "Enter matrix dimensions (m n p) for A[m×n] * B[n×p]: ";
    cin >> m >> n >> p;

    // Thread configuration
    int num_threads;
    cout << "Enter number of threads to use (0 for auto): ";
    cin >> num_threads;
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // Initialize matrices
    vector<vector<int>> A(m, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(p));
    vector<vector<int>> C_seq(m, vector<int>(p));
    vector<vector<int>> C_par(m, vector<int>(p));
    
    srand(time(0));
    initializeMatrix(A, m, n);
    initializeMatrix(B, n, p);

    // Print input matrices if they're small
    if (m <= 10 && n <= 10 && p <= 10) {
        printMatrix(A, "A");
        printMatrix(B, "B");
    }

    // Sequential multiplication
    auto start_seq = high_resolution_clock::now();
    sequentialMultiply(A, B, C_seq, m, n, p);
    auto stop_seq = high_resolution_clock::now();
    auto duration_seq = duration_cast<microseconds>(stop_seq - start_seq);

    // Parallel multiplication
    auto start_par = high_resolution_clock::now();
    parallelMultiply(A, B, C_par, m, n, p);
    auto stop_par = high_resolution_clock::now();
    auto duration_par = duration_cast<microseconds>(stop_par - start_par);

    // Verify results
    bool results_match = true;
    for (int i = 0; i < m && results_match; ++i) {
        for (int j = 0; j < p; ++j) {
            if (C_seq[i][j] != C_par[i][j]) {
                results_match = false;
                break;
            }
        }
    }

    // Print results
    cout << "\nResults:";
    cout << "\nMatrix dimensions: " << m << "x" << n << " * " << n << "x" << p;
    cout << "\nActual threads used: " << omp_get_max_threads();
    cout << "\nSequential time: " << duration_seq.count() << " μs";
    cout << "\nParallel time: " << duration_par.count() << " μs";
    
    // Handle division by zero for speedup calculation
    if (duration_par.count() > 0) {
        double speedup = static_cast<double>(duration_seq.count()) / duration_par.count();
        cout << "\nSpeedup factor: " << fixed << setprecision(2) << speedup << "x";
    } else {
        cout << "\nSpeedup factor: Too fast to measure";
    }
    
    cout << "\nResults match: " << (results_match ? "Yes" : "No") << endl;

    // Print result matrices if they're small
    if (m <= 10 && p <= 10) {
        printMatrix(C_seq, "Sequential Result");
        printMatrix(C_par, "Parallel Result");
    }

    return 0;
}