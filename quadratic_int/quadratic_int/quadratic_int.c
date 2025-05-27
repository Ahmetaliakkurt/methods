/*****************************************************************************
 * Schrödinger's Siths - Kalp Şeklinde Poisson Çözümü
 *   (Paralel Tek Array Jacobi - Thread Sınırlarında 4-Noktalı Quadratic Extrapolasyon)
 *****************************************************************************/

#define _USE_MATH_DEFINES  // M_PI için
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define GRID_SIZE      500
#define MAX_ITERATIONS 100000000
#define TOLERANCE      1e-10

#define X_MIN -1.5
#define X_MAX  1.5
#define Y_MIN -1.5
#define Y_MAX  1.5

static inline double chargeDensity(double x, double y) {
    return sin(x)*cos(y);
}

static inline int isInsideHeart(double x, double y) {
    double e = x * x + y * y - 1.0;
    return (e * e * e - x * x * y * y * y) <= 0.0;
}

int main() {
    int N = GRID_SIZE;
    double dx = (X_MAX - X_MIN) / (N - 1);
    double dy = (Y_MAX - Y_MIN) / (N - 1);
    size_t total = (size_t)N * N;

    // Bellek tahsisi
    double* u_old = malloc(total * sizeof(double));
    double* u_new = malloc(total * sizeof(double));
    int* mask = malloc(total * sizeof(int));
    double* xv = malloc(N * sizeof(double));
    double* yv = malloc(N * sizeof(double));
    if (!u_old || !u_new || !mask || !xv || !yv) {
        fprintf(stderr, "Bellek tahsisi hatası\n");
        return 1;
    }

    // Koordinatlar
    for (int i = 0; i < N; i++) {
        xv[i] = X_MIN + i * dx;
        yv[i] = Y_MIN + i * dy;
    }

    // Başlangıç ve mask
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            size_t idx = i * N + j;
            u_old[idx] = u_new[idx] = 0.0;
            mask[idx] = isInsideHeart(xv[i], yv[j]);
        }
    }

    printf("Paralel Tek‐Array Jacobi (4-Noktalı Quadratic Extrapolasyon) başlıyor...\n");
    printf("GRID_SIZE = %d, TOL = %e, Threads = %d\n",
        N, TOLERANCE, omp_get_max_threads());

    double w0 = omp_get_wtime();
    clock_t c0 = clock();

    int converged = 0;
#pragma omp parallel shared(converged)
    {
        int T = omp_get_num_threads();
        int t = omp_get_thread_num();
        int cols = N / T;
        int j0 = t * cols;
        int j1 = (t == T - 1 ? N - 1 : j0 + cols - 1);

        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            double localMax = 0.0;

            // 1) Thread‐boundary'de 4‐noktalı quadratic extrapolation
#pragma omp for collapse(2)
            for (int i = 0; i < N; ++i) {
                for (int j = j0; j <= j1; ++j) {
                    // üst/alt kenar değil, sadece sol/sağ sınırda uygulama
                    if (j == j0) {
                        // sol sınır: iki iç + iki dış nokta
                        double f_m2 = u_old[i * N + (j0 - 2)];
                        double f_m1 = u_old[i * N + (j0 - 1)];
                        double f_p1 = u_old[i * N + (j0 + 1)];
                        double f_p2 = u_old[i * N + (j0 + 2)];
                        u_old[i * N + j0] =
                            -1.0 / 6.0 * f_m2
                            + 2.0 / 3.0 * f_m1
                            + 2.0 / 3.0 * f_p1
                            - 1.0 / 6.0 * f_p2;
                    }
                    else if (j == j1) {
                        // sağ sınır: iki iç + iki dış
                        double f_m1 = u_old[i * N + (j1 - 1)];
                        double f_m2 = u_old[i * N + (j1 - 2)];
                        double f_p1 = u_old[i * N + (j1 + 1)];
                        double f_p2 = u_old[i * N + (j1 + 2)];
                        u_old[i * N + j1] =
                            -1.0 / 6.0 * f_p2
                            + 2.0 / 3.0 * f_p1
                            + 2.0 / 3.0 * f_m1
                            - 1.0 / 6.0 * f_m2;
                    }
                }
            }
#pragma omp barrier

            // 2) Jacobi güncellemesi ve maxDiff hesabı
#pragma omp for collapse(2) reduction(max:localMax)
            for (int i = 0; i < N; ++i) {
                for (int j = j0; j <= j1; ++j) {
                    size_t idx = i * N + j;
                    if (!mask[idx]) {
                        u_new[idx] = u_old[idx];
                        continue;
                    }
                    // Yukarı / Aşağı komşu
                    double up = (i > 0 ? u_old[(i - 1) * N + j] : u_old[(i + 1) * N + j]);
                    double down = (i < N - 1 ? u_old[(i + 1) * N + j] : u_old[(i - 1) * N + j]);
                    // Sol / Sağ komşu
                    double left = (j > 0 ? u_old[i * N + (j - 1)] : u_old[i * N + (j + 1)]);
                    double right = (j < N - 1 ? u_old[i * N + (j + 1)] : u_old[i * N + (j - 1)]);

                    double rhs = dx * dy * chargeDensity(xv[i], yv[j]);
                    double nv = 0.25 * (up + down + left + right - rhs);
                    u_new[idx] = nv;
                    double df = fabs(nv - u_old[idx]);
                    if (df > localMax) localMax = df;
                }
            }
#pragma omp barrier

            // 3) Konverjans kontrol
#pragma omp single
            {
                if (localMax < TOLERANCE) {
                    printf("Ulaşıldı: iter = %d, maxDiff = %e\n", iter, localMax);
                    converged = 1;
                }
            }
#pragma omp barrier
            if (converged) break;

            // 4) Swap
#pragma omp single
            {
                double* tmp = u_old;
                u_old = u_new;
                u_new = tmp;
            }
#pragma omp barrier
        }
    }

    double w1 = omp_get_wtime() - w0;
    double c1 = (double)(clock() - c0) / CLOCKS_PER_SEC;
    printf("Wall-clock: %.6f s, CPU: %.6f s\n", w1, c1);

    // 5) Sonuçları yaz
    FILE* f = fopen("solution_thread_boundary_quad_ext.dat", "w");
    int cnt = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (mask[i * N + j]) {
                fprintf(f, "%.15f %.15f %.15f\n", xv[i], yv[j], u_old[i * N + j]);
                cnt++;
            }
        }
    }
    fclose(f);
    printf("Yazılan nokta sayısı: %d\n", cnt);

    free(u_old);
    free(u_new);
    free(mask);
    free(xv);
    free(yv);
    return 0;
}
