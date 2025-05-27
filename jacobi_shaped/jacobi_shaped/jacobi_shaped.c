/*****************************************************************************
 * Schrödinger's Siths - Kalp Şeklinde Poisson Çözümü
 * Seri CPU Jacobi – Memory-Safe & Optimize
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define GRID_SIZE      500
#define MAX_ITERATIONS 100000000
#define TOLERANCE      1e-10
#define X_MIN         -1.5
#define X_MAX          1.5
#define Y_MIN         -1.5
#define Y_MAX          1.5

int main(void) {
    size_t N = GRID_SIZE;
    size_t N2 = N * N;
    double dx = (X_MAX - X_MIN) / (N - 1);
    double dy = (Y_MAX - Y_MIN) / (N - 1);
    printf("Jacobi başlıyor (N=%zu, tol=%.1e)...\n", N, TOLERANCE);

    /* 1) Ana diziler */
    double* u_old = malloc(N2 * sizeof * u_old);
    if (!u_old) { fprintf(stderr, "malloc(u_old) failed\n"); return 1; }
    double* u_new = malloc(N2 * sizeof * u_new);
    if (!u_new) { fprintf(stderr, "malloc(u_new) failed\n"); return 1; }
    unsigned char* mask = malloc(N2 * sizeof * mask);
    if (!mask) { fprintf(stderr, "malloc(mask) failed\n");    return 1; }
    double* sinx = malloc(N * sizeof * sinx);
    if (!sinx) { fprintf(stderr, "malloc(sinx) failed\n");    return 1; }
    double* cosy = malloc(N * sizeof * cosy);
    if (!cosy) { fprintf(stderr, "malloc(cosy) failed\n");    return 1; }

    /* 2) Ön-hesaplama: sin(x), cos(y), mask ve başlangıç u=0 */
    size_t insideCount = 0;
    for (size_t i = 0; i < N; ++i) {
        double x = X_MIN + i * dx;
        sinx[i] = sin(x);
        for (size_t j = 0; j < N; ++j) {
            size_t idx = i * N + j;
            u_old[idx] = u_new[idx] = 0.0;
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                mask[idx] = 0;
            }
            else {
                double y = Y_MIN + j * dy;
                double expr = x * x + y * y - 1.0;
                mask[idx] = (expr * expr * expr - x * x * y * y * y <= 0.0);
                if (mask[idx]) insideCount++;
            }
        }
    }
    printf("İç nokta sayısı: %zu\n", insideCount);

    /* 3) Sadece iç noktalar için satır/kol indeksi listesi */
    size_t* rows = malloc(insideCount * sizeof * rows);
    if (!rows) { fprintf(stderr, "malloc(rows) failed\n"); return 1; }
    size_t* cols = malloc(insideCount * sizeof * cols);
    if (!cols) { fprintf(stderr, "malloc(cols) failed\n"); return 1; }

    for (size_t k = 0, i = 1; i < N - 1; ++i) {
        for (size_t j = 1; j < N - 1; ++j) {
            if (mask[i * N + j]) {
                rows[k] = i;
                cols[k] = j;
                ++k;
            }
        }
    }

    /* 4) cos(y) ön-hesaplama */
    for (size_t j = 0; j < N; ++j) {
        double y = Y_MIN + j * dy;
        cosy[j] = cos(y);
    }

    /* 5) Jacobi iterasyonu */
    clock_t t0 = clock();
    double coef = dx * dy;
    size_t iter;
    for (iter = 0; iter < MAX_ITERATIONS; ++iter) {
        double maxDiff = 0.0;
        for (size_t k = 0; k < insideCount; ++k) {
            size_t i = rows[k], j = cols[k];
            size_t id = i * N + j;
            double rhs = coef * (sinx[i] * cosy[j]);
            double up = u_old[id - N];
            double down = u_old[id + N];
            double left = u_old[id - 1];
            double right = u_old[id + 1];
            double newV = 0.25 * (up + down + left + right - rhs);
            u_new[id] = newV;
            double d = newV - u_old[id];
            if (d < 0) d = -d;
            if (d > maxDiff) maxDiff = d;
        }
        if (maxDiff < TOLERANCE) {
            printf("Yakınsama %zu iterasyonda (maxDiff=%.3e)\n", iter, maxDiff);
            break;
        }
        /* swap */
        double* tmp = u_old; u_old = u_new; u_new = tmp;
    }
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("İterasyon: %zu, CPU süresi: %.3f s\n", iter, elapsed);

    /* 6) Sonuçları yaz */
    FILE* out = fopen("solution_shape.dat", "w");
    if (!out) perror("fopen");
    for (size_t k = 0; k < insideCount; ++k) {
        size_t i = rows[k], j = cols[k];
        double xx = X_MIN + i * dx, yy = Y_MIN + j * dy;
        fprintf(out, "%.6f %.6f %.6f\n", xx, yy, u_old[i * N + j]);
    }
    fclose(out);

    /* 7) Bellek temizle */
    free(u_old);
    free(u_new);
    free(mask);
    free(sinx);
    free(cosy);
    free(rows);
    free(cols);

    return 0;
}
