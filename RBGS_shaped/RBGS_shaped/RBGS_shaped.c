/*****************************************************************************
 * Schr�dinger's Siths - Kalp �eklinde Poisson ��z�m� (Seri CPU Red-Black Gauss-Seidel)
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define GRID_SIZE      500
#define MAX_ITERATIONS 100000000
#define TOLERANCE      1e-8

#define X_MIN -1.5
#define X_MAX  1.5
#define Y_MIN -1.5
#define Y_MAX  1.5

 // Poisson denkleminin sa� taraf� i�in fonksiyon: f(x,y) = sin(x)
double chargeDensity(double x, double y) {
    return sin(x)*cos(y);
}

// "Kalp" �eklinin i�inde mi, de�il mi kontrol�
// (x^2 + y^2 - 1)^3 - x^2*y^3 <= 0 ise nokta kalp i�erisindedir.
int isInsideHeart(double x, double y) {
    double expr = (x * x + y * y - 1.0);
    double val = (expr * expr * expr) - (x * x * y * y * y);
    return (val <= 0.0) ? 1 : 0;
}

int main() {
    printf("lets start to calculate for N = %d and tol = %f...\n", GRID_SIZE, TOLERANCE);
    clock_t startTime = clock();

    // Grid ad�m b�y�kl�kleri
    double dx = (X_MAX - X_MIN) / (GRID_SIZE - 1);
    double dy = (Y_MAX - Y_MIN) / (GRID_SIZE - 1);
    int totalSize = GRID_SIZE * GRID_SIZE;

    // Bellek tahsisi (1D diziler �zerinden 2D eri�im; indeks = i * GRID_SIZE + j)
    double* u = (double*)malloc(totalSize * sizeof(double));
    int* mask = (int*)malloc(totalSize * sizeof(int));
    double* xVals = (double*)malloc(GRID_SIZE * sizeof(double));
    double* yVals = (double*)malloc(GRID_SIZE * sizeof(double));
    if (!u || !mask || !xVals || !yVals) {
        fprintf(stderr, "Bellek tahsisinde hata!\n");
        exit(EXIT_FAILURE);
    }

    // x ve y koordinatlar�n� olu�tur
    for (int i = 0; i < GRID_SIZE; i++) {
        xVals[i] = X_MIN + i * dx;
        yVals[i] = Y_MIN + i * dy;
    }

    // Kalp maskesi ve ba�lang�� ��z�m� (u=0) ayar�
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            int idx = i * GRID_SIZE + j;
            mask[idx] = isInsideHeart(xVals[i], yVals[j]);
            u[idx] = 0.0;
        }
    }

    int iter;
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        double maxDiff = 0.0;
        double diff;

        // Red ad�m: (i+j) �ift olan noktalar
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {
                if ((i + j) % 2 == 0) {
                    int idx = i * GRID_SIZE + j;
                    if (mask[idx] == 1) {
                        double oldVal = u[idx];
                        double rhs = dx * dy * chargeDensity(xVals[i], yVals[j]);
                        double up = u[(i - 1) * GRID_SIZE + j];
                        double down = u[(i + 1) * GRID_SIZE + j];
                        double left = u[i * GRID_SIZE + (j - 1)];
                        double right = u[i * GRID_SIZE + (j + 1)];
                        double newVal = 0.25 * (up + down + left + right - rhs);
                        u[idx] = newVal;
                        diff = fabs(newVal - oldVal);
                        if (diff > maxDiff)
                            maxDiff = diff;
                    }
                }
            }
        }

        // Black ad�m: (i+j) tek olan noktalar
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {
                if ((i + j) % 2 == 1) {
                    int idx = i * GRID_SIZE + j;
                    if (mask[idx] == 1) {
                        double oldVal = u[idx];
                        double rhs = dx * dy * chargeDensity(xVals[i], yVals[j]);
                        double up = u[(i - 1) * GRID_SIZE + j];
                        double down = u[(i + 1) * GRID_SIZE + j];
                        double left = u[i * GRID_SIZE + (j - 1)];
                        double right = u[i * GRID_SIZE + (j + 1)];
                        double newVal = 0.25 * (up + down + left + right - rhs);
                        u[idx] = newVal;
                        diff = fabs(newVal - oldVal);
                        if (diff > maxDiff)
                            maxDiff = diff;
                    }
                }
            }
        }

        if (maxDiff < TOLERANCE) {
            printf("��z�m %d iterasyonda ula��ld� (maxDiff = %.6e).\n", iter, maxDiff);
            break;
        }
        // �ste�e ba�l�: Belirli aral�klarla iterasyon bilgisini yazd�rabilirsiniz.
    }

    if (iter == MAX_ITERATIONS) {
        printf("��z�m %d iterasyonda yak�nsama sa�lamad�.\n", MAX_ITERATIONS);
    }

    // Sonu�lar� sadece kalp i�i noktalar i�in dosyaya yaz
    FILE* outputFile = fopen("solution_shape.dat", "w");
    if (!outputFile) {
        perror("Dosya a��lamad�");
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
    printf("��z�m 'solution_shape.dat' dosyas�na yaz�ld�. Yaz�lan nokta say�s� = %d\n", countWritten);

    double elapsedTime = (double)(clock() - startTime) / CLOCKS_PER_SEC;
    printf("�al��ma s�resi: %.6f saniye\n", elapsedTime);

    // Bellek temizli�i
    free(u);
    free(mask);
    free(xVals);
    free(yVals);

    return 0;
}
