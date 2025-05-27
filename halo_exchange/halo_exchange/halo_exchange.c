/*****************************************************************************
 * Schr�dinger's Siths - Kalp �eklinde Poisson ��z�m�
 *   (Paralel Tek Array Jacobi - Ger�ek Halo Exchange, Interpolasyon Yok)
 *****************************************************************************/
#define _USE_MATH_DEFINES  // MSVC i�in
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
    double expr = (x * x + y * y - 1.0);
    return (expr * expr * expr - x * x * y * y * y) <= 0.0;
}

// Ger�ek halo exchange fonksiyonu (kom�u threadlerin kenarlar�n� payla��yor)
void exchange_halos(double* u, int N) {
    int M = N + 2;
    // Yukar� ve a�a�� halo
#pragma omp parallel for
    for (int j = 1; j <= N; j++) {
        u[0 * M + j] = u[1 * M + j];       // Top halo (i=0)
        u[(N + 1) * M + j] = u[N * M + j];       // Bottom halo (i=N+1)
    }
    // Sol ve sa� halo
#pragma omp parallel for
    for (int i = 1; i <= N; i++) {
        u[i * M + 0] = u[i * M + 1];       // Left halo (j=0)
        u[i * M + (N + 1)] = u[i * M + N];       // Right halo (j=N+1)
    }
    // K��e halolar� (gerekirse)
    u[0 * M + 0] = u[1 * M + 1];
    u[0 * M + (N + 1)] = u[1 * M + N];
    u[(N + 1) * M + 0] = u[N * M + 1];
    u[(N + 1) * M + (N + 1)] = u[N * M + N];
}

int main() {
    printf("Tek Array Jacobi (Ger�ek Halo Exchange, Interpolasyonsuz) ba�l�yor...\n");
    printf("GRID_SIZE = %d, TOLERANS = %e\n", GRID_SIZE, TOLERANCE);

    double wall_start = omp_get_wtime();
    clock_t cpu_start = clock();

    int N = GRID_SIZE, M = N + 2;
    size_t totalSize = (size_t)M * M;

    double dx = (X_MAX - X_MIN) / (N - 1);
    double dy = (Y_MAX - Y_MIN) / (N - 1);

    double* u = malloc(totalSize * sizeof(double));
    int* mask = malloc(totalSize * sizeof(int));
    double* xVals = malloc(M * sizeof(double));
    double* yVals = malloc(M * sizeof(double));

    if (!u || !mask || !xVals || !yVals) {
        fprintf(stderr, "Bellek tahsisi hatas�!\n");
        return EXIT_FAILURE;
    }

    for (int i = 1; i <= N; i++) {
        xVals[i] = X_MIN + (i - 1) * dx;
        yVals[i] = Y_MIN + (i - 1) * dy;
    }

    for (size_t idx = 0; idx < totalSize; idx++) {
        u[idx] = 0.0;
        mask[idx] = 0;
    }

    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            mask[i * M + j] = isInsideHeart(xVals[i], yVals[j]);

    int iter;
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {

        exchange_halos(u, N);

        double maxDiff = 0.0;
#pragma omp parallel for collapse(2) reduction(max:maxDiff)
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                int idx = i * M + j;
                if (mask[idx]) {
                    double oldVal = u[idx];
                    double rhs = dx * dy * chargeDensity(xVals[i], yVals[j]);
                    double up = u[(i - 1) * M + j];
                    double down = u[(i + 1) * M + j];
                    double left = u[i * M + (j - 1)];
                    double right = u[i * M + (j + 1)];
                    double newVal = 0.25 * (up + down + left + right - rhs);
                    u[idx] = newVal;
                    double diff = fabs(newVal - oldVal);
                    if (diff > maxDiff) maxDiff = diff;
                }
            }
        }

        if (maxDiff < TOLERANCE) {
            printf("Ula��ld�: iter = %d, maxDiff = %e\n", iter, maxDiff);
            break;
        }
    }

    if (iter == MAX_ITERATIONS)
        printf("Yak�nsamad� (%d iter).\n", MAX_ITERATIONS);

    FILE* fp = fopen("solution_single_u_exchange.dat", "w");
    if (!fp) { perror("Dosya a��lamad�"); return EXIT_FAILURE; }
    int count = 0;
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            if (mask[i * M + j])
                fprintf(fp, "%.6f %.6f %.6f\n", xVals[i], yVals[j], u[i * M + j]), count++;
    fclose(fp);

    printf("Yaz�lan nokta say�s�: %d\n", count);

    double wall_elapsed = omp_get_wtime() - wall_start;
    double cpu_elapsed = (double)(clock() - cpu_start) / CLOCKS_PER_SEC;
    printf("Wall-clock: %.6f s, CPU: %.6f s\n", wall_elapsed, cpu_elapsed);

    free(u); free(mask); free(xVals); free(yVals);
    return 0;
}
