/*****************************************************************************
 * Schr�dinger's Siths - Kalp �eklinde Poisson ��z�m� (Paralel Jacobi - �yile�tirilmi�, Clock �l��m� ve Ba�lang�� Tahmini Dosyadan Okuma)
 *****************************************************************************/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define GRID_SIZE      500
#define MAX_ITERATIONS 100000000
#define TOLERANCE      1e-8

#define X_MIN -2.5
#define X_MAX  2.5
#define Y_MIN -2.5
#define Y_MAX  2.5

 // Poisson denkleminin sa� taraf�: f(x,y) = sin(x)*cos(y)
double chargeDensity(double x, double y) {
    return sin(x);
}

// "Kalp" �eklinin i�inde mi kontrol�
int isInsideHeart(double x, double y) {
    double expr = (x * x + y * y - 1.0);
    double val = (expr * expr * expr) - (x * x * y * y * y);
    return (val <= 0.0);
}

int main() {
    printf("Jacobi hesaplamas� ba�l�yor. GRID_SIZE=%d, Tolerans=%e\n", GRID_SIZE, TOLERANCE);

    // Zaman �l��mleri
    double wall_start = omp_get_wtime();
    clock_t cpu_start = clock();

    // Grid ad�mlar�
    double dx = (X_MAX - X_MIN) / (GRID_SIZE - 1);
    double dy = (Y_MAX - Y_MIN) / (GRID_SIZE - 1);
    int totalSize = GRID_SIZE * GRID_SIZE;

    // Bellek tahsisi
    double* u_old = malloc(totalSize * sizeof(double));
    double* u_new = malloc(totalSize * sizeof(double));
    int* mask = malloc(totalSize * sizeof(int));
    double* xVals = malloc(GRID_SIZE * sizeof(double));
    double* yVals = malloc(GRID_SIZE * sizeof(double));
    if (!u_old || !u_new || !mask || !xVals || !yVals) {
        fprintf(stderr, "Bellek tahsisinde hata!\n");
        return EXIT_FAILURE;
    }

    // x, y koordinatlar�n�n olu�turulmas�
    for (int i = 0; i < GRID_SIZE; i++) {
        xVals[i] = X_MIN + i * dx;
        yVals[i] = Y_MIN + i * dy;
    }

    // Ba�lang��: mask ve s�f�r doldurma
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            int idx = i * GRID_SIZE + j;
            mask[idx] = isInsideHeart(xVals[i], yVals[j]);
            u_old[idx] = 0.0;
            u_new[idx] = 0.0;
        }
    }

    // --- Ba�lang�� tahminini dosyadan oku ---
    // Windows tam yol, ters b�l� ka���l�
    const char* initPath = "C:\\Users\\ahmet\\source\\repos\\Jacobi_Shaped_CPUparalel\\Jacobi_Shaped_CPUparalel\\solution_jacobi_cpu.dat";
    FILE* finit = fopen(initPath, "r");
    if (finit) {
        double x_in, y_in, u_in;
        while (fscanf(finit, "%lf %lf %lf", &x_in, &y_in, &u_in) == 3) {
            int i = (int)round((x_in - X_MIN) / dx);
            int j = (int)round((y_in - Y_MIN) / dy);
            if (i >= 0 && i < GRID_SIZE && j >= 0 && j < GRID_SIZE) {
                int idx = i * GRID_SIZE + j;
                if (mask[idx]) {
                    u_old[idx] = u_in;  // Ba�lang�� de�erini ata
                    u_new[idx] = u_in;
                }
            }
        }
        fclose(finit);
        printf("Ba�lang�� tahmini '%s' dosyas�ndan y�klendi.\n", initPath);
    }
    else {
        printf("Uyar�: '%s' bulunamad�. S�f�rla ba�lat�l�yor.\n", initPath);
    }

    // Jacobi iterasyonlar�
    int iter;
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        double maxDiff = 0.0;
#pragma omp parallel for collapse(2) reduction(max:maxDiff)
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {
                int idx = i * GRID_SIZE + j;
                if (mask[idx]) {
                    double oldVal = u_old[idx];
                    double rhs = dx * dy * chargeDensity(xVals[i], yVals[j]);
                    double up = u_old[idx - GRID_SIZE];
                    double down = u_old[idx + GRID_SIZE];
                    double left = u_old[idx - 1];
                    double right = u_old[idx + 1];
                    double newVal = 0.25 * (up + down + left + right - rhs);
                    u_new[idx] = newVal;
                    double diff = fabs(newVal - oldVal);
                    if (diff > maxDiff) maxDiff = diff;
                }
                else {
                    u_new[idx] = u_old[idx];
                }
            }
        }
        if (maxDiff < TOLERANCE) {
            printf("��z�m %d iterasyonda sa�land� (maxDiff=%e).\n", iter, maxDiff);
            break;
        }
        // Pointer swap
        double* tmp = u_old; u_old = u_new; u_new = tmp;
    }
    if (iter == MAX_ITERATIONS) {
        printf("%d iterasyonda yak�nsama sa�lanamad�.\n", MAX_ITERATIONS);
    }

    // Sonu�lar� dosyaya yaz
    FILE* fout = fopen("solution_jacobi_cpu.dat", "w");
    if (!fout) { perror("Dosya a��lamad�"); return EXIT_FAILURE; }
    int count = 0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            int idx = i * GRID_SIZE + j;
            if (mask[idx]) {
                fprintf(fout, "%.6f %.6f %.6f\n", xVals[i], yVals[j], u_old[idx]);
                count++;
            }
        }
    }
    fclose(fout);
    printf("%d nokta 'solution_jacobi_cpu.dat' dosyas�na yaz�ld�.\n", count);

    // Zaman �l��mleri
    double wall_elapsed = omp_get_wtime() - wall_start;
    double cpu_elapsed = (double)(clock() - cpu_start) / CLOCKS_PER_SEC;
    printf("Wall-clock s�re: %.6f s\n", wall_elapsed);
    printf("CPU s�resi: %.6f s\n", cpu_elapsed);

    // Bellek temizli�i
    free(u_old); free(u_new); free(mask); free(xVals); free(yVals);
    return 0;
}