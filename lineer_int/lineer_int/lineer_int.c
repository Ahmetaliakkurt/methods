/*****************************************************************************
 * Schrödinger's Siths - Kalp Şeklinde Poisson Çözümü
 *   (Paralel Tek Array Jacobi - 1D Lineer İnterpolasyon, Thread Sınırlarında,
 *    OpenMP Reduction ile MaxDiff Düzeltmesi)
 *****************************************************************************/

#define _USE_MATH_DEFINES
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
    double expr = x * x + y * y - 1.0;
    return (expr * expr * expr - x * x * y * y * y) <= 0.0;
}

int main() {
    printf("Tek Array Jacobi (Thread Sınırlarında 1D Lineer İnterpolasyon) başlıyor...\n");
    printf("GRID_SIZE = %d, TOLERANS = %e, Threads = %d\n", GRID_SIZE, TOLERANCE, omp_get_max_threads());

    int N = GRID_SIZE, M = N + 2;
    size_t totalSize = (size_t)M * M;
    double dx = (X_MAX - X_MIN) / (N - 1);
    double dy = (Y_MAX - Y_MIN) / (N - 1);

    // Bellek tahsisi
    double* u_old = malloc(totalSize * sizeof(double));
    double* u_new = malloc(totalSize * sizeof(double));
    int* mask = malloc(totalSize * sizeof(int));
    double* xVals = malloc(M * sizeof(double));
    double* yVals = malloc(M * sizeof(double));
    if (!u_old || !u_new || !mask || !xVals || !yVals) {
        fprintf(stderr, "Bellek tahsisi hatası!\n");
        return EXIT_FAILURE;
    }

    // Koordinatları hazırla
    for (int i = 1; i <= N; ++i) {
        xVals[i] = X_MIN + (i - 1) * dx;
        yVals[i] = Y_MIN + (i - 1) * dy;
    }
    // Başlangıç değerleri
    for (size_t k = 0; k < totalSize; ++k) {
        u_old[k] = u_new[k] = 0.0;
        mask[k] = 0;
    }
    for (int i = 1; i <= N; ++i)
        for (int j = 1; j <= N; ++j)
            mask[i * M + j] = isInsideHeart(xVals[i], yVals[j]);

    double wall_start = omp_get_wtime();
    clock_t cpu_start = clock();

    // Domain decomposition by columns
#pragma omp parallel
    {
        int T = omp_get_num_threads();
        int t = omp_get_thread_num();
        int cols_per = N / T;
        int j0 = t * cols_per + 1;
        int j1 = (t == T - 1) ? N : j0 + cols_per - 1;

        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            // Interpolasyon: thread sınırlarında komşudan veri almadan lineer ortalama
            // Üst ve alt sınırlar
#pragma omp for
            for (int j = j0; j <= j1; ++j) {
                // i=1
                u_old[1 * M + j] = 0.5 * (u_old[2 * M + j] + u_old[3 * M + j]);
                // i=N
                u_old[N * M + j] = 0.5 * (u_old[(N - 1) * M + j] + u_old[(N - 2) * M + j]);
            }
            // Sol ve sağ sınırlar
#pragma omp for
            for (int i = 1; i <= N; ++i) {
                u_old[i * M + j0] = 0.5 * (u_old[i * M + j0 + 1] + u_old[i * M + j0 + 2]);
                u_old[i * M + j1] = 0.5 * (u_old[i * M + j1 - 1] + u_old[i * M + j1 - 2]);
            }
#pragma omp barrier

            // Jacobi güncellemesi ve azami farkın hesaplanması, reduction ile
            double maxDiff = 0.0;
#pragma omp for collapse(2) reduction(max:maxDiff)
            for (int i = 1; i <= N; ++i) {
                for (int j = j0; j <= j1; ++j) {
                    int idx = i * M + j;
                    if (!mask[idx]) { u_new[idx] = u_old[idx]; continue; }
                    double up = u_old[(i > 1 ? i - 1 : i + 1) * M + j];
                    double down = u_old[(i < N ? i + 1 : i - 1) * M + j];
                    double left = u_old[i * M + (j > 1 ? j - 1 : j + 1)];
                    double right = u_old[i * M + (j < N ? j + 1 : j - 1)];
                    double rhs = dx * dy * chargeDensity(xVals[i], yVals[j]);
                    double newv = 0.25 * (up + down + left + right - rhs);
                    u_new[idx] = newv;
                    double diff = fabs(newv - u_old[idx]);
                    if (diff > maxDiff) maxDiff = diff;
                }
            }
#pragma omp barrier

            // Yakınsamayı kontrol et
#pragma omp single
            {
                if (maxDiff < TOLERANCE) {
                    printf("Ulaşıldı: iter = %d, maxDiff = %e\n", iter, maxDiff);
                    iter = MAX_ITERATIONS;
                }
            }
#pragma omp barrier
            if (maxDiff < TOLERANCE) break;

            // swap pointer
#pragma omp single
            {
                double* tmp = u_old; u_old = u_new; u_new = tmp;
            }
#pragma omp barrier
        }
    }

    double wall_elapsed = omp_get_wtime() - wall_start;
    double cpu_elapsed = (double)(clock() - cpu_start) / CLOCKS_PER_SEC;
    printf("Wall-clock: %.6f s, CPU: %.6f s\n", wall_elapsed, cpu_elapsed);

    // Sonuçları yaz
    FILE* fp = fopen("solution_thread_boundary_interp.dat", "w");
    int count = 0;
    for (int i = 1; i <= N; ++i)
        for (int j = 1; j <= N; ++j)
            if (mask[i * M + j])
                fprintf(fp, "%.15f %.15f %.15f\n", xVals[i], yVals[j], u_old[i * M + j]), ++count;
    fclose(fp);

    free(u_old); free(u_new);
    free(mask);
    free(xVals); free(yVals);
    return 0;
}
