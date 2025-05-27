/*****************************************************************************
 * Schrödinger's Siths - Kalp Şeklinde Poisson Çözümü
 *   (Paralel Tek Array Jacobi - Thread Sınırlarında 1D Ekstrapolasyon, Köşelerde Ortalama)
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
    return sin(x) * cos(y);
}

static inline int isInsideHeart(double x, double y) {
    double expr = x * x + y * y - 1.0;
    return (expr * expr * expr - x * x * y * y * y) <= 0.0;
}

int main() {
    printf("Paralel Tek-Array Jacobi (Ekstrapolasyon Thread Sınırlarında) başlıyor...\n");
    printf("GRID_SIZE = %d, TOL = %e, Threads = %d\n",
        GRID_SIZE, TOLERANCE, omp_get_max_threads());

    int N = GRID_SIZE;
    int M = N + 2;
    size_t total = (size_t)M * M;
    double dx = (X_MAX - X_MIN) / (N - 1);
    double dy = (Y_MAX - Y_MIN) / (N - 1);

    // çift tampon
    double* u_old = malloc(total * sizeof * u_old);
    double* u_new = malloc(total * sizeof * u_new);
    int* mask = malloc(total * sizeof * mask);
    double* xv = malloc(M * sizeof * xv);
    double* yv = malloc(M * sizeof * yv);
    if (!u_old || !u_new || !mask || !xv || !yv) {
        fprintf(stderr, "Bellek tahsisi hatası!\n");
        return EXIT_FAILURE;
    }

    // Koordinatları hazırla
    for (int i = 1; i <= N; i++) {
        xv[i] = X_MIN + (i - 1) * dx;
        yv[i] = Y_MIN + (i - 1) * dy;
    }
    // Başlangıç değerleri ve mask
    for (size_t k = 0; k < total; k++) {
        u_old[k] = u_new[k] = 0.0;
        mask[k] = 0;
    }
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            mask[i * M + j] = isInsideHeart(xv[i], yv[j]);
        }
    }

    double wall0 = omp_get_wtime();
    clock_t cpu0 = clock();

    // Paralel domain decomposition
#pragma omp parallel
    {
        int T = omp_get_num_threads();
        int t = omp_get_thread_num();
        int cols = N / T;
        int j0 = t * cols + 1;
        int j1 = (t == T - 1 ? N : j0 + cols - 1);

        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            // --- Ekstrapolasyon thread sınırlarında ---
#pragma omp for
            for (int j = j0; j <= j1; ++j) {
                // üst gerçek hücre i=1
                u_old[1 * M + j] = 2 * u_old[2 * M + j] - u_old[3 * M + j];
                // alt gerçek hücre i=N
                u_old[N * M + j] = 2 * u_old[(N - 1) * M + j] - u_old[(N - 2) * M + j];
            }
#pragma omp for
            for (int i = 1; i <= N; ++i) {
                // sol sınır hücresi j=j0
                u_old[i * M + j0] = 2 * u_old[i * M + j0 + 1] - u_old[i * M + j0 + 2];
                // sağ sınır hücresi j=j1
                u_old[i * M + j1] = 2 * u_old[i * M + j1 - 1] - u_old[i * M + j1 - 2];
            }
            // köşelerde basit ortalama
#pragma omp single
            {
                u_old[1 * M + j0] = 0.5 * (u_old[1 * M + j0 + 1] + u_old[2 * M + j0]);
                u_old[1 * M + j1] = 0.5 * (u_old[1 * M + j1 - 1] + u_old[2 * M + j1]);
                u_old[N * M + j0] = 0.5 * (u_old[N * M + j0 + 1] + u_old[(N - 1) * M + j0]);
                u_old[N * M + j1] = 0.5 * (u_old[N * M + j1 - 1] + u_old[(N - 1) * M + j1]);
            }
#pragma omp barrier

            // --- Jacobi güncellemesi + reduction ile maxDiff hesapla ---
            double maxDiff = 0.0;
#pragma omp for collapse(2) reduction(max:maxDiff)
            for (int i = 1; i <= N; i++) {
                for (int j = j0; j <= j1; j++) {
                    int idx = i * M + j;
                    if (!mask[idx]) {
                        u_new[idx] = u_old[idx];
                        continue;
                    }
                    double up = u_old[(i > 1 ? i - 1 : i + 1) * M + j];
                    double down = u_old[(i < N ? i + 1 : i - 1) * M + j];
                    double left = u_old[i * M + (j > 1 ? j - 1 : j + 1)];
                    double right = u_old[i * M + (j < N ? j + 1 : j - 1)];
                    double rhs = dx * dy * chargeDensity(xv[i], yv[j]);
                    double val = 0.25 * (up + down + left + right - rhs);
                    u_new[idx] = val;
                    double df = fabs(val - u_old[idx]);
                    if (df > maxDiff) maxDiff = df;
                }
            }
#pragma omp barrier

            // --- Konverjans kontrol ve swap ---
#pragma omp single
            {
                if (maxDiff < TOLERANCE) {
                    printf("Ulaşıldı: iter = %d, maxDiff = %e\n", iter, maxDiff);
                    iter = MAX_ITERATIONS;
                }
            }
#pragma omp barrier
            if (maxDiff < TOLERANCE) break;

#pragma omp single
            { double* tmp = u_old; u_old = u_new; u_new = tmp; }
#pragma omp barrier
        }
    }

    double wall1 = omp_get_wtime() - wall0;
    double cpu1 = (double)(clock() - cpu0) / CLOCKS_PER_SEC;
    printf("Wall-clock: %.6f s, CPU: %.6f s\n", wall1, cpu1);

    // sonuçları dosyaya yaz
    FILE* fp = fopen("solution_thread_boundary_extrap.dat", "w");
    int count = 0;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            if (mask[i * M + j]) {
                fprintf(fp, "%.12f %.12f %.12f\n", xv[i], yv[j], u_old[i * M + j]);
                ++count;
            }
        }
    }
    fclose(fp);
    printf("Yazılan nokta sayısı: %d\n", count);

    free(u_old);
    free(u_new);
    free(mask);
    free(xv);
    free(yv);
    return 0;
}
