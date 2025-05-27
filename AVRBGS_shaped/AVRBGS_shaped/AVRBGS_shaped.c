/*****************************************************************************
 * Schr�dinger's Siths - Kalp �eklinde Poisson ��z�m�
 * (Seri CPU Red-Black Gauss-Seidel + Her 1000 iterasyonda grid smoothing)
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define GRID_SIZE      500
#define MAX_ITERATIONS 100000000
#define TOLERANCE      1e-8
#define AVG_ITER       5000


#define X_MIN -1.5
#define X_MAX  1.5
#define Y_MIN -1.5
#define Y_MAX  1.5

double chargeDensity(double x, double y) {
    return sin(x)*cos(y);
}

int isInsideHeart(double x, double y) {
    double expr = (x * x + y * y - 1.0);
    double val = (expr * expr * expr) - (x * x * y * y * y);
    return (val <= 0.0) ? 1 : 0;
}

void smoothGrid(double* u, int gridSize, const int* mask) {
    double* temp = (double*)malloc(gridSize * gridSize * sizeof(double));
    if (!temp) {
        fprintf(stderr, "Bellek tahsisinde hata (smoothGrid)!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < gridSize * gridSize; i++) {
        temp[i] = u[i];
    }

    for (int i = 1; i < gridSize - 1; i++) {
        for (int j = 1; j < gridSize - 1; j++) {
            int idx = i * gridSize + j;
            if (mask[idx] == 1) {
                double sum = u[idx];
                int count = 1;
                // �st kom�u
                if (mask[(i - 1) * gridSize + j] == 1) {
                    sum += u[(i - 1) * gridSize + j];
                    count++;
                }
                // Alt kom�u
                if (mask[(i + 1) * gridSize + j] == 1) {
                    sum += u[(i + 1) * gridSize + j];
                    count++;
                }
                // Sol kom�u
                if (mask[i * gridSize + (j - 1)] == 1) {
                    sum += u[i * gridSize + (j - 1)];
                    count++;
                }
                // Sa� kom�u
                if (mask[i * gridSize + (j + 1)] == 1) {
                    sum += u[i * gridSize + (j + 1)];
                    count++;
                }
                temp[idx] = sum / count;
            }
        }
    }

    for (int i = 0; i < gridSize * gridSize; i++) {
        u[i] = temp[i];
    }
    free(temp);
}

int main() {
    clock_t startTime = clock();

    double dx = (X_MAX - X_MIN) / (GRID_SIZE - 1);
    double dy = (Y_MAX - Y_MIN) / (GRID_SIZE - 1);
    int totalSize = GRID_SIZE * GRID_SIZE;


    double* u = (double*)malloc(totalSize * sizeof(double));
    int* mask = (int*)malloc(totalSize * sizeof(int));
    double* xVals = (double*)malloc(GRID_SIZE * sizeof(double));
    double* yVals = (double*)malloc(GRID_SIZE * sizeof(double));
    if (!u || !mask || !xVals || !yVals) {
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
            u[idx] = 0.0;
        }
    }

    printf("Starting to calculate for %d step size...\n", AVG_ITER);

    int iter;
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        double maxDiff = 0.0;
        double diff;


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


        if ((iter + 1) % AVG_ITER == 0) {
            smoothGrid(u, GRID_SIZE, mask);
        }
    }

    if (iter == MAX_ITERATIONS) {
        printf("��z�m %d iterasyonda yak�nsama sa�lamad�.\n", MAX_ITERATIONS);
    }

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

    free(u);
    free(mask);
    free(xVals);
    free(yVals);

    return 0;
}
