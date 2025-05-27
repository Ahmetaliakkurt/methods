/*****************************************************************************
 * Schrödinger's Siths - Kalp Şeklinde Poisson Çözümü
 *          (Seri CPU Steepest Descent)
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define GRID_SIZE      500
#define MAX_ITERATIONS 100000000
#define TOLERANCE      1e-10

#define X_MIN -1.5
#define X_MAX  1.5
#define Y_MIN -1.5
#define Y_MAX  1.5

 // f = sin(x)*cos(y)
double chargeDensity(double x, double y) {
    return sin(x) * cos(y);
}

// Kalp maskesi
int isInsideHeart(double x, double y) {
    double expr = (x * x + y * y - 1.0);
    double val = expr * expr * expr - x * x * y * y * y;
    return (val <= 0.0) ? 1 : 0;
}

// A·u işlemi (5-point Laplace)
void applyA(const double* u, double* Au, const int* mask, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            if (!mask[idx]) { Au[idx] = 0.0; continue; }
            double s = 0.0;
            if (i > 0 && mask[idx - N]) Au[idx] += u[idx - N];
            if (i < N - 1 && mask[idx + N]) Au[idx] += u[idx + N];
            if (j > 0 && mask[idx - 1]) Au[idx] += u[idx - 1];
            if (j < N - 1 && mask[idx + 1]) Au[idx] += u[idx + 1];
            Au[idx] -= 4.0 * u[idx];
        }
    }
}

// İç çarpım
double dot(const double* v1, const double* v2, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v1[i] * v2[i];
    return s;
}

int main() {
    clock_t t0 = clock();
    int N = GRID_SIZE, M = N * N;

    double dx = (X_MAX - X_MIN) / (N - 1), dy = (Y_MAX - Y_MIN) / (N - 1);

    // Hafıza ayırma
    double* u = calloc(M, sizeof(double));
    double* b = malloc(M * sizeof(double));
    double* r = malloc(M * sizeof(double));
    double* Ar = malloc(M * sizeof(double));
    int* mask = malloc(M * sizeof(int));
    double* xv = malloc(N * sizeof(double)), * yv = malloc(N * sizeof(double));

    // Grid noktaları
    for (int i = 0;i < N;i++) {
        xv[i] = X_MIN + i * dx;
        yv[i] = Y_MIN + i * dy;
    }
    // b ve mask
    for (int i = 0;i < N;i++) for (int j = 0;j < N;j++) {
        int idx = i * N + j;
        mask[idx] = isInsideHeart(xv[i], yv[j]);
        b[idx] = mask[idx] ? dx * dy * chargeDensity(xv[i], yv[j]) : 0.0;
    }

    // Başlangıç: u=0 ⇒ r=b-Au = b
    for (int i = 0;i < M;i++) {
        u[i] = 0.0;
        r[i] = b[i];
    }

    double rnorm2 = dot(r, r, M), tol2 = TOLERANCE * TOLERANCE;
    int iter;
    for (iter = 0; iter<MAX_ITERATIONS && rnorm2>tol2; iter++) {
        // Ar = A·r
        applyA(r, Ar, mask, N);

        double alpha = rnorm2 / dot(r, Ar, M);

        // u_{k+1} = u_k + α r_k
        // r_{k+1} = r_k - α A r_k
        for (int i = 0;i < M;i++) {
            u[i] += alpha * r[i];
            r[i] -= alpha * Ar[i];
        }

        // Yeni residual normu
        rnorm2 = dot(r, r, M);
        if (iter % 1000 == 0) {
            printf("Iter %6d  ||r|| = %.3e\n", iter, sqrt(rnorm2));
        }
    }

    printf("Converged at iter %d, ||r|| = %.3e\n", iter, sqrt(rnorm2));

    // Çözümü kaydet
    FILE* f = fopen("solution_shape_sd.dat", "w");
    for (int i = 0;i < N;i++) for (int j = 0;j < N;j++) {
        int idx = i * N + j;
        if (mask[idx]) fprintf(f, "%.6f %.6f %.6f\n", xv[i], yv[j], u[idx]);
    }
    fclose(f);

    double elapsed = (clock() - t0) / (double)CLOCKS_PER_SEC;
    printf("Time: %.3f s\n", elapsed);

    // Bellek temizliği
    free(u); free(b); free(r);
    free(Ar); free(mask); free(xv); free(yv);
    return 0;
}
