#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NX 1000
#define NY 1000
#define N (NX * NY)
#define TOL 1e-7
#define MAX_ITER 10000000
#define PI 3.14159265358979323846

// 5 point stencil
void stencil(const double* x, double* Ax) {
    int ix, iy, idx;
    for (iy = 0; iy < NY; iy++) {
        for (ix = 0; ix < NX; ix++) {
            idx = iy * NX + ix;
            double center = 4.0 * x[idx];
            double left = (ix > 0) ? x[idx - 1] : 0.0;
            double right = (ix < NX - 1) ? x[idx + 1] : 0.0;
            double up = (iy > 0) ? x[idx - NX] : 0.0;
            double down = (iy < NY - 1) ? x[idx + NX] : 0.0;
            Ax[idx] = center - left - right - up - down;
        }
    }
}

// dot product
double dot(const double* x, const double* y, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

int main() {
    int i, iter;
    double alpha, beta, rsold, rsnew;

    double* x = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* r = (double*)malloc(N * sizeof(double));
    double* p = (double*)malloc(N * sizeof(double));
    double* Ap = (double*)malloc(N * sizeof(double));

    if (x == NULL || b == NULL || r == NULL || p == NULL || Ap == NULL) {
        printf("Bellek tahsisinde hata!\n");
        exit(1);
    }

    // step size
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);

    // initial values and source
    for (i = 0; i < N; i++) {
        x[i] = 0.0;
        int ix = i % NX;
        int iy = i / NX;
        double x_coord = ix * dx;
        double y_coord = iy * dy;
        b[i] = sin(PI * x_coord) * sin(PI * y_coord);
    }

    // residual
    for (i = 0; i < N; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }
    rsold = dot(r, r, N);
    printf("calculating the fckn poisson...\n\n");
    clock_t start = clock();


    for (iter = 0; iter < MAX_ITER; iter++) {
        stencil(p, Ap);
        alpha = rsold / dot(p, Ap, N);
        for (i = 0; i < N; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        rsnew = dot(r, r, N);
        if (sqrt(rsnew) < TOL) {
            iter++;
            break;
        }
        beta = rsnew / rsold;

        for (i = 0; i < N; i++) {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Solved %d iterations \nfinal residual: %e\n", iter, sqrt(rsnew));
    printf("execution time: %f s\n", time_spent);


    free(x);
    free(b);
    free(r);
    free(p);
    free(Ap);

    return 0;
}
