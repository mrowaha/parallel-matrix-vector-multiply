/**
 * a serial implementation of matrix vector multiplication
 * @author: Muhammad Rowaha
 */
#include<stdlib.h>
#include<string.h>

#include"matrix-ops.h"
#include"spd-loader.h"
#include"config.h"

int main(int argc, char** argv) {

    double *A, *x;
    const int n = N;

    A = load_matrix(n);
    x = load_vector(n);

    double *r = malloc(sizeof(double) * n);
    matrix_vector_multiply(A, x, r, n, n);

    char filename[100] = "serial_mvm.txt";
    write_vector_to_file(r, n, filename);

    free(A);
    free(x);
    free(r);
    return EXIT_SUCCESS;
}