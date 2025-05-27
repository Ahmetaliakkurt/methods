/*****************************************************************************
 * Schr�dinger's Siths - Kalp �eklinde Poisson ��z�m� (Seri CPU Conjugate Gradient)
 * Jacobi Preconditioner Eklenmi� Versiyon
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define GRID_SIZE      1000
#define MAX_ITERATIONS 100000000
#define TOLERANCE      1e-10

#define X_MIN -2.5
#define X_MAX  2.5
#define Y_MIN -2.5
#define Y_MAX  2.5

double chargeDensity(double x, double y) {
    return sin(x);
}

int isInsideHeart(double x, double y) {
    double expr = (x * x + y * y - 1.0);
    double val = (expr * expr * expr) - (x * x * y * y * y);
    return (val <= 0.0) ? 1 : 0;
}

void applyA(const double* u, double* Au, const int* mask, int gridSize) {
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            int idx = i * gridSize + j;
            if (mask[idx] == 1) {
                double sum = 0.0;
                // �st kom�u
                if (i > 0 && mask[(i - 1) * gridSize + j] == 1)
                    sum += u[(i - 1) * gridSize + j];
                // Alt kom�u
                if (i < gridSize - 1 && mask[(i + 1) * gridSize + j] == 1)
                    sum += u[(i + 1) * gridSize + j];
                // Sol kom�u
                if (j > 0 && mask[i * gridSize + (j - 1)] == 1)
                    sum += u[i * gridSize + (j - 1)];
                // Sa� kom�u
                if (j < gridSize - 1 && mask[i * gridSize + (j + 1)] == 1)
                    sum += u[i * gridSize + (j + 1)];
                Au[idx] = sum - 4.0 * u[idx];
            }
            else {
                Au[idx] = 0.0;
            }
        }
    }
}

double dotProduct(const double* v1, const double* v2, int totalSize) {
    double sum = 0.0;
    for (int i = 0; i < totalSize; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

int main() {
    clock_t startTime = clock();

    double dx = (X_MAX - X_MIN) / (GRID_SIZE - 1);
    double dy = (Y_MAX - Y_MIN) / (GRID_SIZE - 1);
    int totalSize = GRID_SIZE * GRID_SIZE;

    double* u = (double*)calloc(totalSize, sizeof(double));
    double* b = (double*)malloc(totalSize * sizeof(double));
    double* r = (double*)malloc(totalSize * sizeof(double));
    double* z = (double*)malloc(totalSize * sizeof(double)); // Preconditioned residual
    double* p = (double*)malloc(totalSize * sizeof(double));
    double* Ap = (double*)malloc(totalSize * sizeof(double));
    int* mask = (int*)malloc(totalSize * sizeof(int));
    double* xVals = (double*)malloc(GRID_SIZE * sizeof(double));
    double* yVals = (double*)malloc(GRID_SIZE * sizeof(double));

    if (!u || !b || !r || !z || !p || !Ap || !mask || !xVals || !yVals) {
        fprintf(stderr, "Bellek tahsisinde hata!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < GRID_SIZE; i++) {
        xVals[i] = X_MIN + i * dx;
        yVals[i] = Y_MIN + i * dy;
    }

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            int idx = i * GRID_SIZE + j;
            mask[idx] = isInsideHeart(xVals[i], yVals[j]);
            if (mask[idx] == 1) {
                b[idx] = dx * dy * chargeDensity(xVals[i], yVals[j]);
            }
            else {
                b[idx] = 0.0;
            }
        }
    }

    // Ba�lang��ta: r = b
    for (int i = 0; i < totalSize; i++) {
        r[i] = b[i];
        // Jacobi preconditioner: M = diag(A); burada mask i�indeki noktalar i�in A'n�n diagonal eleman� -4't�r.
        if (mask[i] == 1) {
            z[i] = r[i] / (-4.0);
        }
        else {
            z[i] = 0.0;
        }
        // �lk p vekt�r� preconditioned residual ile ba�lar.
        p[i] = z[i];
    }

    // r^T * z de�eri hesaplan�yor
    double rz_old = dotProduct(r, z, totalSize);

    int iter;
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        applyA(p, Ap, mask, GRID_SIZE);

        double pAp = dotProduct(p, Ap, totalSize);
        if (pAp == 0.0) {
            printf("p^T*A*p s�f�r oldu, algoritma durduruluyor.\n");
            break;
        }
        double alpha = rz_old / pAp;

        for (int i = 0; i < totalSize; i++) {
            u[i] = u[i] + alpha * p[i];
            r[i] = r[i] - alpha * Ap[i];
        }

        // Jacobi preconditioner uygulanarak z = M^{-1}r hesaplan�r.
        for (int i = 0; i < totalSize; i++) {
            if (mask[i] == 1) {
                z[i] = r[i] / (-4.0);
            }
            else {
                z[i] = 0.0;
            }
        }

        double rz_new = dotProduct(r, z, totalSize);
        if (iter % 1000 == 0) {
            printf("Iteration: %d, residual = %.6e\n", iter, sqrt(dotProduct(r, r, totalSize)));
        }
        if (sqrt(dotProduct(r, r, totalSize)) < TOLERANCE) {
            printf("Iteration: %d\nresidual = %.6e.\n", iter, sqrt(dotProduct(r, r, totalSize)));
            break;
        }

        double beta = rz_new / rz_old;
        for (int i = 0; i < totalSize; i++) {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }

    if (iter == MAX_ITERATIONS) {
        printf("Not converged on %d iterations.\n", MAX_ITERATIONS);
    }

    // Sadece kalp i�indeki noktalar�n ��z�m�n� dosyaya yaz
    FILE* outputFile = fopen("solution_shape.dat", "w");
    if (!outputFile) {
        perror("��z�m dosyas� a��lamad�");
        exit(EXIT_FAILURE);
    }
    int countWritten = 0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            int idx = i * GRID_SIZE + j;
            if (mask[idx] == 1) {
                fprintf(outputFile, "%.6f %.6f %.6f\n", xVals[i], yVals[j], u[idx]);
                countWritten++;
            }
        }
    }
    fclose(outputFile);
    printf("Solution saved on 'solution_shape.dat'\nwritten point = %d\n", countWritten);

    double elapsedTime = (double)(clock() - startTime) / CLOCKS_PER_SEC;
    printf("Execution time: %.6f s\n", elapsedTime);

    free(u);
    free(b);
    free(r);
    free(z);
    free(p);
    free(Ap);
    free(mask);
    free(xVals);
    free(yVals);

    return 0;
}
