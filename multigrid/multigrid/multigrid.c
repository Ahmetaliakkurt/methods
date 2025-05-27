/*****************************************************************************
 * Schrödinger's Siths - Kalp Şeklinde Poisson Çözümü
 * V-Cycle Multigrid – Düzenli Grid ve Artırılmış Smoothing
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_FINE        512      // Grid noktası: 2^k + 1 için N = N_FINE + 1 = 513
#define MAX_VCYCLES   200      // Maksimum V-cycle tekrarı
#define TOL           1e-6     // Rezidü normu toleransı
#define PI            3.141592653589793

 // --- Dyn. bellek tahsisi / serbest bırakma -----------------------------------
double** allocate_grid(int n) {
    double** A = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        A[i] = calloc(n, sizeof(double));
        if (!A[i]) { perror("calloc"); exit(1); }
    }
    return A;
}

void free_grid(double** A, int n) {
    for (int i = 0; i < n; i++) free(A[i]);
    free(A);
}

// --- Gauss–Seidel rahatlatıcı ------------------------------------------------
void gauss_seidel(double** u, double** f, int n, int nu) {
    double h2 = 1.0 / ((n - 1) * (n - 1));
    for (int sweep = 0; sweep < nu; sweep++) {
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                u[i][j] = 0.25 * (h2 * f[i][j]
                    + u[i - 1][j] + u[i + 1][j]
                    + u[i][j - 1] + u[i][j + 1]);
            }
        }
    }
}

// --- Rezidü hesaplama --------------------------------------------------------
void compute_residual(double** u, double** f, double** r, int n) {
    double h2 = 1.0 / ((n - 1) * (n - 1));
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            r[i][j] = f[i][j]
                - (4 * u[i][j]
                    - u[i - 1][j] - u[i + 1][j]
                    - u[i][j - 1] - u[i][j + 1]) * h2;
        }
    }
}

// --- L2-norm hesaplama ------------------------------------------------------
double residual_norm(double** r, int n) {
    double sum = 0.0;
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            sum += r[i][j] * r[i][j];
        }
    }
    return sqrt(sum);
}

// --- Full-weighting Restriction (fine -> coarse) ----------------------------
void restrict_full(double** r_f, double** f_c, int n_f) {
    int n_c = (n_f - 1) / 2 + 1;
    for (int I = 1; I < n_c - 1; I++) {
        for (int J = 1; J < n_c - 1; J++) {
            int i = 2 * I, j = 2 * J;
            f_c[I][J] = (
                4.0 * r_f[i][j]
                + 2.0 * (r_f[i - 1][j] + r_f[i + 1][j] + r_f[i][j - 1] + r_f[i][j + 1])
                + (r_f[i - 1][j - 1] + r_f[i - 1][j + 1]
                    + r_f[i + 1][j - 1] + r_f[i + 1][j + 1])
                ) / 16.0;
        }
    }
}

// --- Bilinear prolongation & ekleme (coarse -> fine) ------------------------
void prolongate_and_add(double** u_c, double** u_f, int n_f) {
    int n_c = (n_f - 1) / 2 + 1;
    for (int I = 0; I < n_c; I++)
        for (int J = 0; J < n_c; J++)
            u_f[2 * I][2 * J] += u_c[I][J];
    for (int I = 0; I < n_c - 1; I++)
        for (int J = 0; J < n_c; J++)
            u_f[2 * I + 1][2 * J] += 0.5 * (u_c[I][J] + u_c[I + 1][J]);
    for (int I = 0; I < n_c; I++)
        for (int J = 0; J < n_c - 1; J++)
            u_f[2 * I][2 * J + 1] += 0.5 * (u_c[I][J] + u_c[I][J + 1]);
    for (int I = 0; I < n_c - 1; I++)
        for (int J = 0; J < n_c - 1; J++)
            u_f[2 * I + 1][2 * J + 1] += 0.25 * (
                u_c[I][J] + u_c[I + 1][J]
                + u_c[I][J + 1] + u_c[I + 1][J + 1]
                );
}

// --- V-Cycle Multigrid -------------------------------------------------------
void v_cycle(double** u, double** f, int n) {
    const int nu1 = 4, nu2 = 4;       // Pre / post smoothing sweeps arttırıldı
    const int n_coarse_min = 3;
    if (n <= n_coarse_min) {
        gauss_seidel(u, f, n, 50);
        return;
    }
    gauss_seidel(u, f, n, nu1);
    double** res = allocate_grid(n);
    compute_residual(u, f, res, n);
    int n_c = (n - 1) / 2 + 1;
    double** f_c = allocate_grid(n_c);
    restrict_full(res, f_c, n);
    double** u_c = allocate_grid(n_c);
    v_cycle(u_c, f_c, n_c);
    prolongate_and_add(u_c, u, n);
    gauss_seidel(u, f, n, nu2);
    free_grid(res, n);
    free_grid(f_c, n_c);
    free_grid(u_c, n_c);
}

int main() {
    int N = N_FINE + 1;  // 513 = 2^9 + 1
    double h = 1.0 / (N - 1);
    double** u = allocate_grid(N);
    double** f = allocate_grid(N);
    double** res;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = i * h, y = j * h;
            f[i][j] = sin(4 * PI * x) * sin(PI * y);
        }
    }

    clock_t t0 = clock();
    double err;
    size_t cycle;
    for (cycle = 0; cycle < MAX_VCYCLES; cycle++) {
        v_cycle(u, f, N);
        res = allocate_grid(N);
        compute_residual(u, f, res, N);
        err = residual_norm(res, N);
        free_grid(res, N);
        printf("V-cycle %3zu: ||r||_2 = %.3e\n", cycle + 1, err);
        if (err < TOL) break;
    }
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Tamamlandı: %zu V-cycle sonra, toplam süre = %.3f s\n", cycle + 1, elapsed);

    // çözüm örneklemesi...
    printf("Çözüm (köşe 9×9 örnek):\n");
    int step = (N - 1) / 8;
    for (int i = 0; i < N; i += step) {
        for (int j = 0; j < N; j += step) {
            printf("%6.3f ", u[i][j]);
        }
        printf("\n");
    }
    free_grid(u, N);
    free_grid(f, N);
    return 0;
}
