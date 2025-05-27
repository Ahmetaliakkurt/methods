/*****************************************************************************
 * Schr�dinger's Siths - Kalp �eklinde Poisson ��z�m� (Seri CPU Conjugate Gradient)
 * Incomplete Cholesky Preconditioner Eklenmi� Versiyon
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

 // Fonksiyonlar (�rnekteki charge density ve kalp i�i kontrol�)
double chargeDensity(double x, double y) {
    return sin(x)*cos(y);
}

int isInsideHeart(double x, double y) {
    double expr = (x * x + y * y - 1.0);
    double val = (expr * expr * expr) - (x * x * y * y * y);
    return (val <= 0.0) ? 1 : 0;
}

// Uygulanan A operat�r�: 5 noktal� stencil (u'nun kom�u de�erleri - 4*u)
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
                // A operat�r�m�z: A u = (kom�ular�n toplam�) - 4*u
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
   Incomplete Cholesky i�in;
   B = -A olarak kabul edersek, aktif noktalar i�in:
       B(i,i) = 4.0,
       B(i,j) = -1.0 (kom�u ba�lant�lar)
   Lexikografik s�rada, bir noktaya ait off-diagonal elemanlar yaln�zca sol ve �st kom�ulardan gelecektir.
   Bu �rnekte, sadece diagonal fakt�r L_diag hesaplan�p,
   off-diagonaller analitik olarak: L_off_left = -1.0 / L_diag(left), L_off_top = -1.0 / L_diag(top) �eklinde al�nacakt�r.
*/
void computeIncompleteCholesky(double* L_diag, const int* mask, int gridSize) {
    int totalSize = gridSize * gridSize;
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            int idx = i * gridSize + j;
            if (mask[idx] == 1) {
                double sum = 0.0;
                // E�er sol kom�u varsa
                if (j > 0 && mask[idx - 1] == 1) {
                    // Off-diagonal eleman: L(i,j) = -1 / L_diag(i, j-1)
                    double L_left = -1.0 / L_diag[idx - 1];
                    sum += L_left * L_left;
                }
                // E�er �st kom�u varsa
                if (i > 0 && mask[idx - gridSize] == 1) {
                    double L_top = -1.0 / L_diag[idx - gridSize];
                    sum += L_top * L_top;
                }
                // B(i,i) = 4.0
                double diag_val = 4.0 - sum;
                if (diag_val <= 0) {
                    fprintf(stderr, "Incomplete Cholesky: negatıf (i=%d, j=%d)!\n", i, j);
                    exit(EXIT_FAILURE);
                }
                L_diag[idx] = sqrt(diag_val);
            }
            else {
                L_diag[idx] = 0.0;
            }
        }
    }
}

// Forward yerine koyma: L * y = r
// Lexikografik s�rayla (sat�r sat�r)
void forwardSolve(const double* r, double* y, const double* L_diag, const int* mask, int gridSize) {
    int totalSize = gridSize * gridSize;
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            int idx = i * gridSize + j;
            if (mask[idx] == 1) {
                double temp = r[idx];
                // Sol kom�u
                if (j > 0 && mask[idx - 1] == 1) {
                    // L(i,j) = -1 / L_diag(idx-1)
                    temp -= (-1.0 / L_diag[idx - 1]) * y[idx - 1];
                }
                // �st kom�u
                if (i > 0 && mask[idx - gridSize] == 1) {
                    temp -= (-1.0 / L_diag[idx - gridSize]) * y[idx - gridSize];
                }
                y[idx] = temp / L_diag[idx];
            }
            else {
                y[idx] = 0.0;
            }
        }
    }
}

// Backward yerine koyma: L^T * z = y
// Ters y�nde (sondan ba�a) ��z�m
void backwardSolve(const double* y, double* z, const double* L_diag, const int* mask, int gridSize) {
    int totalSize = gridSize * gridSize;
    // Ba�ta t�m z elemanlar�n� s�f�rla
    for (int i = 0; i < totalSize; i++) {
        z[i] = 0.0;
    }
    for (int i = gridSize - 1; i >= 0; i--) {
        for (int j = gridSize - 1; j >= 0; j--) {
            int idx = i * gridSize + j;
            if (mask[idx] == 1) {
                double temp = y[idx];
                // Sa� kom�u: (i, j+1)
                if (j < gridSize - 1 && mask[idx + 1] == 1) {
                    // L(i, j+1) = -1 / L_diag(idx) (ayn� L eleman�, ��nk� (i,j+1) i�in sol kom�u = (i,j))
                    temp -= (-1.0 / L_diag[idx]) * z[idx + 1];
                }
                // Alt kom�u: (i+1, j)
                if (i < gridSize - 1 && mask[idx + gridSize] == 1) {
                    temp -= (-1.0 / L_diag[idx]) * z[idx + gridSize];
                }
                z[idx] = temp / L_diag[idx];
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
    double* y = (double*)malloc(totalSize * sizeof(double)); // ara ��z�m i�in
    double* z = (double*)malloc(totalSize * sizeof(double)); // preconditioned residual
    double* p = (double*)malloc(totalSize * sizeof(double));
    double* Ap = (double*)malloc(totalSize * sizeof(double));
    int* mask = (int*)malloc(totalSize * sizeof(int));
    double* L_diag = (double*)malloc(totalSize * sizeof(double)); // Incomplete Cholesky i�in diagonal elemanlar
    double* xVals = (double*)malloc(GRID_SIZE * sizeof(double));
    double* yVals = (double*)malloc(GRID_SIZE * sizeof(double));

    if (!u || !b || !r || !y || !z || !p || !Ap || !mask || !L_diag || !xVals || !yVals) {
        fprintf(stderr, "Bellek tahsisinde hata!\n");
        exit(EXIT_FAILURE);
    }

    // Grid noktalar�n� hesapla
    for (int i = 0; i < GRID_SIZE; i++) {
        xVals[i] = X_MIN + i * dx;
        yVals[i] = Y_MIN + i * dy;
    }

    // Mask ve sa� taraf vekt�r�n� (b) olu�tur: yaln�zca kalp i�indeki noktalar aktif
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

    // Preconditioner i�in: B = -A oldu�undan, aktif noktalarda B(i,i)=4.0.
    // L_diag precomputasyonu (lexikografik s�ra ile)
    for (int i = 0; i < totalSize; i++) {
        L_diag[i] = 0.0;
    }
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            int idx = i * GRID_SIZE + j;
            if (mask[idx] == 1) {
                double sum = 0.0;
                if (j > 0 && mask[idx - 1] == 1) {
                    double L_left = -1.0 / L_diag[idx - 1];
                    sum += L_left * L_left;
                }
                if (i > 0 && mask[idx - GRID_SIZE] == 1) {
                    double L_top = -1.0 / L_diag[idx - GRID_SIZE];
                    sum += L_top * L_top;
                }
                double diag_val = 4.0 - sum;
                if (diag_val <= 0) {
                    printf("Incomplete Cholesky: negative or zero pivot found (i=%d, j=%d)!\n", i, j);
                    exit(EXIT_FAILURE);
                }
                L_diag[idx] = sqrt(diag_val);
            }
        }
    }

    // Ba�lang��: u = 0, r = b
    for (int i = 0; i < totalSize; i++) {
        r[i] = b[i];
    }
    // Preconditioner uygulanarak z = M^{-1}r: �nce forward sonra backward ��z�m
    forwardSolve(r, y, L_diag, mask, GRID_SIZE);
    backwardSolve(y, z, L_diag, mask, GRID_SIZE);

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
            printf("p^T*A*p oldu, algorıtma durduruluyor.\n");
            break;
        }
        double alpha = rz_old / pAp;
        for (int i = 0; i < totalSize; i++) {
            u[i] = u[i] + alpha * p[i];
            r[i] = r[i] - alpha * Ap[i];
        }
        // Preconditioner: z = M^{-1}r (forward/backward yerine koyma)
        forwardSolve(r, y, L_diag, mask, GRID_SIZE);
        backwardSolve(y, z, L_diag, mask, GRID_SIZE);

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

    // ��z�m� yaln�zca kalp i�indeki noktalardan dosyaya yaz
    FILE* outputFile = NULL;
    errno_t err = fopen_s(&outputFile, "solution_shape.dat", "w");
    if (err != 0) {
        perror("Dosya açılamadı");
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
    free(y);
    free(z);
    free(p);
    free(Ap);
    free(mask);
    free(L_diag);
    free(xVals);
    free(yVals);

    return 0;
}
