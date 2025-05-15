#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <string.h>

#include"config.h"
#include"spd-loader.h"
#include"matrix-ops.h"

#define MASTER 0

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *A, *x;
    const int n = N;
    const int p = size - 1; // the -np param will be p + 1 because master does not participate in the computations
    const int sqrt_p = (int)sqrt(p); // p is perfect sqrt in this restriction
    const int block_size = n / sqrt_p; // each block in this 2D partitioning will be n / sqrt(p) by n / sqrt(p)
    int coords[2];
    if (sqrt_p * sqrt_p != p) {
        if (rank == MASTER) {
            fprintf(stderr, "Number of workers must be a perfect square\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (n % p != 0) {
        if (rank == MASTER) {
            fprintf(stderr, "N mod workers should be 0");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }


    MPI_Comm WORKER_COMM;
    int color = (rank == MASTER) ? MPI_UNDEFINED : 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &WORKER_COMM);

    if (WORKER_COMM != MPI_COMM_NULL) {
        // only the p worker processes will enter this phase
        int worker_comm_rank, worker_comm_size;
        MPI_Comm_rank(WORKER_COMM, &worker_comm_rank);
        MPI_Comm_size(WORKER_COMM, &worker_comm_size);

        coords[0] = worker_comm_rank / sqrt_p;
        coords[1] = worker_comm_rank % sqrt_p;

        double *A_block = malloc(block_size * block_size * sizeof(double));
        MPI_Recv(A_block, block_size * block_size, MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double *x_block = malloc(block_size * sizeof(double));
        MPI_Recv(x_block, block_size, MPI_DOUBLE, MASTER, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // stage 1: calculate z_{alpha, beta}
        double* z_block = malloc(block_size  * sizeof(double));
        matrix_vector_multiply(A_block, x_block, z_block, block_size, block_size);

        // stage 2: Perform the fold operation
        // After the loop, z_block will hold y^{αβ}
        int log_sqrt_p = (int)(log2(sqrt_p));  // Number of folding steps
        int alpha = coords[0];
        int beta  = coords[1];


        int z_block_size = block_size;
        for (int i = 0; i < log_sqrt_p; i++) {
            int half_size = z_block_size / 2;
            double* z1 = z_block;                  // First half
            double* z2 = z_block + half_size;      // Second half

            int bit_mask = 1 << i;
            int partner_beta = beta ^ bit_mask;    // Flip i-th bit of β
            int partner_rank = alpha * sqrt_p + partner_beta;
        

            MPI_Status status;
            double* recv_buf = malloc(half_size * sizeof(double));
            double* next_zblock = malloc(half_size * sizeof(double));
            if ((beta & bit_mask) != 0) {
                // i-th bit is 1: send z1, receive w2, update z2 := z2 + w2
                MPI_Send(z1, half_size, MPI_DOUBLE, partner_rank, 100 + i, WORKER_COMM);
                MPI_Recv(recv_buf, half_size, MPI_DOUBLE, partner_rank, 200 + i, WORKER_COMM, &status);
                for (int j = 0; j < half_size; ++j) {
                    next_zblock[j] = z2[j] + recv_buf[j];
                }
            } else {
                // i-th bit is 0: receive w1, send z2, update z1 := z1 + w1
                MPI_Recv(recv_buf, half_size, MPI_DOUBLE, partner_rank, 100 + i, WORKER_COMM, &status);
                MPI_Send(z2, half_size, MPI_DOUBLE, partner_rank, 200 + i, WORKER_COMM);
                for (int j = 0; j < half_size; ++j) {
                    next_zblock[j] = z1[j] + recv_buf[j];
                }
            }
            free(z_block);
            free(recv_buf);

            z_block = next_zblock;
            z_block_size = half_size;
        }
        const int final_zblock_size = n / p;
        if (z_block_size != final_zblock_size) {
            fprintf(stderr, "final z size, i.e. y_{alpha, beta} was expected to be n / p = %d but got %d\n", final_zblock_size, z_block_size);
            MPI_Abort(WORKER_COMM, 1);
        }
        // At this point of the fold operation, z_block is now y^{αβ} a subvector of size n / p that contains entries from the fully summed vector

        // stage 3: do a transpose
        int transpose_partner = beta * sqrt_p + alpha;  // (β, α)
        double* transposed_block = malloc(z_block_size * sizeof(double)); // this is now y^{βα}
        if (worker_comm_rank != transpose_partner) {
            MPI_Sendrecv(
                z_block, z_block_size, MPI_DOUBLE, transpose_partner, 0,
                transposed_block, z_block_size, MPI_DOUBLE, transpose_partner, 0,
                WORKER_COMM, MPI_STATUS_IGNORE
            );
        } else {
            memcpy(transposed_block, z_block, z_block_size * sizeof(double));
        }

        // stage 4: expand
        int transposed_block_size = z_block_size; // initially have this much size per process this will increase gradually because of concatenation
        for(int i = log_sqrt_p-1; i >= 0; i--) {
            int bit_mask = 1 << i;
            int partner_alpha = alpha ^ bit_mask;    // Flip i-th bit of β
            int partner_rank = partner_alpha * sqrt_p + beta;
            double* recv_buffer = malloc(transposed_block_size * sizeof(double));
            MPI_Sendrecv(
                transposed_block, transposed_block_size, MPI_DOUBLE, partner_rank, 0,
                recv_buffer, transposed_block_size, MPI_DOUBLE, partner_rank, 0,
                WORKER_COMM, MPI_STATUS_IGNORE
            );

            double* concatenated = NULL;
            if ((alpha & bit_mask) != 0) {
                concatenated = concatenate_vectors(recv_buffer, transposed_block_size, transposed_block, transposed_block_size);
            } else {
                concatenated = concatenate_vectors(transposed_block, transposed_block_size, recv_buffer, transposed_block_size);
            }
            free(transposed_block);
            free(recv_buffer);
            transposed_block = concatenated;
            transposed_block_size *= 2;
        }

        // y_{beta} is now the transposed block
        // now we are going to traverse the first row, i.e. alpha == 0 and send the transposed block to the MASTER for accumulation of the result
        if (alpha == 0) {
            MPI_Send(transposed_block, transposed_block_size, MPI_DOUBLE, MASTER, beta, MPI_COMM_WORLD);
        }

        // Cleanup
        free(A_block);
        free(x_block);
        free(z_block);
        free(transposed_block);
    }

    // distribute A and x to the workers of size p
    if (rank == MASTER) {
        A = load_matrix(n);
        x = load_vector(n);
        // Send matrix and vector parts to each process
        for (int a = 0; a < sqrt_p; ++a) {
            for (int b = 0; b < sqrt_p; ++b) {
                int dest = 1 + a * sqrt_p + b;
                double *A_block = malloc(block_size * block_size * sizeof(double));
                for (int i = 0; i < block_size; ++i) {
                    for (int j = 0; j < block_size; ++j) {
                        A_block[i * block_size + j] =
                            A[(a * block_size + i) * n + (b * block_size + j)];
                    }
                }
                MPI_Send(A_block, block_size * block_size, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
                free(A_block);

                double *x_block = malloc(block_size * sizeof(double));
                for (int j = 0; j < block_size; ++j) {
                    x_block[j] = x[b * block_size + j];
                }
                MPI_Send(x_block, block_size, MPI_DOUBLE, dest, 2, MPI_COMM_WORLD);
                free(x_block);
            }
        }

        // wait for results from the workers of the first row i.e alpha = 0
        double* final_result = malloc(n * sizeof(double)); // n = full vector size
        for (int beta = 0; beta < sqrt_p; beta++) {
            MPI_Recv(final_result + beta * block_size, block_size, MPI_DOUBLE, 
                    MPI_ANY_SOURCE, beta, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        write_vector_to_file(final_result, n, "parallel_mvm.txt");
        free(A);
        free(x);
    }

    MPI_Finalize();
}