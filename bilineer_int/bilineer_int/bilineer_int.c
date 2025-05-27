/*****************************************************************************
 * Schr�dinger's Siths - Kalp �eklinde Poisson ��z�m�
 *   (Paralel Tek Array Jacobi - Thread S�n�rlar�nda 2D Bilineer �nterpolasyon)
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
    return  sin(x) * cos(y);
}
static inline int isInsideHeart(double x, double y) {
    double expr = x * x + y * y - 1.0;
    return (expr * expr * expr - x * x * y * y * y) <= 0.0;
}

// Thread s�n�rlar�nda 2D bilineer interpolasyon
void interpolate_thread_boundaries(double* u, int N, int j0, int j1) {
    int M = N + 2;
    // �st ve alt global s�n�rlar
#pragma omp for
    for (int j = j0; j <= j1; ++j) {
        // i=1 h�cresi (�st ger�ek)
        u[1 * M + j] = 0.25 * (
            u[1 * M + (j - 1)] + u[1 * M + (j + 1)] +
            u[2 * M + (j - 1)] + u[2 * M + (j + 1)]
            );
        // i=N h�cresi (alt ger�ek)
        u[N * M + j] = 0.25 * (
            u[N * M + (j - 1)] + u[N * M + (j + 1)] +
            u[(N - 1) * M + (j - 1)] + u[(N - 1) * M + (j + 1)]
            );
    }
    // sol ve sa� thread s�n�rlar�
#pragma omp for
    for (int i = 1; i <= N; ++i) {
        // j = j0
        u[i * M + j0] = 0.25 * (
            u[(i - 1) * M + j0] + u[(i + 1) * M + j0] +
            u[(i - 1) * M + j0 + 1] + u[(i + 1) * M + j0 + 1]
            );
        // j = j1
        u[i * M + j1] = 0.25 * (
            u[(i - 1) * M + j1] + u[(i + 1) * M + j1] +
            u[(i - 1) * M + j1 - 1] + u[(i + 1) * M + j1 - 1]
            );
    }
#pragma omp barrier
}

int main() {
    printf("Tek Array Jacobi (Thread S�n�rlar�nda 2D Bilineer �nterpolasyon) ba�l�yor...\n");
    printf("GRID_SIZE = %d, TOLERANS = %e, Threads = %d\n", GRID_SIZE, TOLERANCE, omp_get_max_threads());

    int N = GRID_SIZE, M = N + 2;
    size_t totalSize = (size_t)M * M;
    double dx = (X_MAX - X_MIN) / (N - 1);
    double dy = (Y_MAX - Y_MIN) / (N - 1);

    double* u_old = malloc(totalSize * sizeof(double));
    double* u_new = malloc(totalSize * sizeof(double));
    int* mask = malloc(totalSize * sizeof(int));
    double* xVals = malloc(M * sizeof(double));
    double* yVals = malloc(M * sizeof(double));
    if (!u_old || !u_new || !mask || !xVals || !yVals) {
        fprintf(stderr, "Bellek tahsisi hatas�!\n");
        return EXIT_FAILURE;
    }
    // koordinatlar
    for (int i = 1; i <= N; i++) {
        xVals[i] = X_MIN + (i - 1) * dx;
        yVals[i] = Y_MIN + (i - 1) * dy;
    }
    for (size_t k = 0; k < totalSize; k++) {
        u_old[k] = u_new[k] = 0.0;
        mask[k] = 0;
    }
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            mask[i * M + j] = isInsideHeart(xVals[i], yVals[j]);

    double wall0 = omp_get_wtime();
    clock_t cpu0 = clock();

#pragma omp parallel
    {
        int T = omp_get_num_threads();
        int t = omp_get_thread_num();
        int cols_per = N / T;
        int j0 = t * cols_per + 1;
        int j1 = (t == T - 1) ? N : j0 + cols_per - 1;

        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            // 2D bilineer interpolation at boundaries
            interpolate_thread_boundaries(u_old, N, j0, j1);

            // Jacobi update with reduction
            double maxDiff = 0.0;
#pragma omp for collapse(2) reduction(max:maxDiff)
            for (int i = 1; i <= N; i++) {
                for (int j = j0; j <= j1; j++) {
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
#pragma omp single
            {
                if (maxDiff < TOLERANCE) {
                    printf("Ula��ld�: iter = %d, maxDiff = %e\n", iter, maxDiff);
                    iter = MAX_ITERATIONS;
                }
            }
#pragma omp barrier
            if (maxDiff < TOLERANCE) break;

            // swap buffers
#pragma omp single
            { double* tmp = u_old; u_old = u_new; u_new = tmp; }
#pragma omp barrier
        }
    }

    double wall1 = omp_get_wtime() - wall0;
    double cpu1 = (double)(clock() - cpu0) / CLOCKS_PER_SEC;
    printf("Wall-clock: %.6f s, CPU: %.6f s\n", wall1, cpu1);

    FILE* fp = fopen("solution_thread_boundary_bilinear.dat", "w");
    int c = 0;
    for (int i = 1; i <= N; i++) for (int j = 1; j <= N; j++) if (mask[i * M + j])
        fprintf(fp, "%.12f %.12f %.12f\n", xVals[i], yVals[j], u_old[i * M + j]), ++c;
    fclose(fp);

    free(u_old); free(u_new); free(mask); free(xVals); free(yVals);
    return 0;
}
