/*****************************************************************************
 * Schr�dinger's Siths - Kalp �eklinde Poisson ��z�m� (Seri CPU Conjugate Gradient)
 * Incomplete Poisson (IP) Preconditioner Eklenmi� Versiyon
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

 // Fonksiyonlar: charge density ve kalp �eklini belirleyen maske fonksiyonu.
double chargeDensity(double x, double y) {
    return sin(x)*cos(y);
}

int isInsideHeart(double x, double y) {
    double expr = (x * x + y * y - 1.0);
    double val = (expr * expr * expr) - (x * x * y * y * y);
    return (val <= 0.0) ? 1 : 0;
}

// A operat�r�n� uygular: 5 noktal� stencil; aktif noktalarda
// Au = (kom�ular�n toplam�) - 4*u
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

// Dot product hesaplamas�
double dotProduct(const double* v1, const double* v2, int totalSize) {
    double sum = 0.0;
    for (int i = 0; i < totalSize; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

/*
   Incomplete Poisson (IP) Preconditioner Uygulamas�:

   Makaledeki IP preconditioner, A = D + L + L^T �eklindeki
   ayr���m� temel al�r. Burada aktif noktalarda D = -4 ve
   D^{-1} = -1/4. Yakla��k ters, Neumann serisinin ilk
   terimleriyle ��yle uygulanabilir:

       z = -0.25 * r + 0.0625 * (r_sol + r_sa� + r_�st + r_alt)

   Bu fonksiyon, yaln�zca aktif (mask==1) grid noktalar� i�in
   uygulan�r. Kenar ve inaktif noktalar i�in z = 0.
*/
void applyIP(const double* r, double* z, const int* mask, int gridSize) {
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            int idx = i * gridSize + j;
            if (mask[idx] == 1) {
                double sum_neighbors = 0.0;
                if (j > 0 && mask[idx - 1] == 1)
                    sum_neighbors += r[idx - 1];
                if (j < gridSize - 1 && mask[idx + 1] == 1)
                    sum_neighbors += r[idx + 1];
                if (i > 0 && mask[idx - gridSize] == 1)
                    sum_neighbors += r[idx - gridSize];
                if (i < gridSize - 1 && mask[idx + gridSize] == 1)
                    sum_neighbors += r[idx + gridSize];
                z[idx] = -0.25 * r[idx] + 0.0625 * sum_neighbors;
            }
            else {
                z[idx] = 0.0;
            }
        }
    }
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

    // Grid noktalar�n� hesapla
    for (int i = 0; i < GRID_SIZE; i++) {
        xVals[i] = X_MIN + i * dx;
        yVals[i] = Y_MIN + i * dy;
    }

    // Mask ve sa� taraf vekt�r�n� (b) olu�tur: sadece kalp i�indeki noktalar aktif
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

    // Ba�lang��: u = 0, r = b
    for (int i = 0; i < totalSize; i++) {
        u[i] = 0.0;
        r[i] = b[i];
    }
    // IP preconditioner uygulanarak z = M_IP^{-1}r
    applyIP(r, z, mask, GRID_SIZE);

    // �lk p vekt�r�: preconditioned residual
    for (int i = 0; i < totalSize; i++) {
        p[i] = z[i];
    }
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
        // IP preconditioner: z = M_IP^{-1}r
        applyIP(r, z, mask, GRID_SIZE);
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
