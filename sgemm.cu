#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float tmp = 0.f;
            for (int k = 0; k < K; k++) {
                tmp += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = tmp;
        }
    }
}


__global__ void mynaiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
        int n = blockDim.x*blockIdx.x+threadIdx.x;
        int m = blockDim.y*blockIdx.y+threadIdx.y;

        if (n<N && m<M){
            float tmp = 0.f;
            for (int k=0;k<K;++k){
                tmp+=a[OFFSET(m,k,K)]*b[OFFSET(k,n,N)];
            }
            c[OFFSET(m,n,N)]=tmp;
        }


    }

__global__ void mySgemm_tiled(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
        const int BM = 128;
        const int BN = 128;
        const int BK = 8;
        const int TM = 8;
        const int TN = 8;

        int n_block = blockIdx.x;
        int m_block = blockIdx.y;
        int tid = blockDim.x*threadIdx.y+threadIdx.x;
        int tid_x=threadIdx.x;
        int tid_y=threadIdx.y;

        __shared__ float shr_a[BM*BK];
        __shared__ float shr_b[BN*BK];
        float reg_c[TM*TN]={0.f};

        int load_a_smem_m = tid >> 1;
        int load_a_smem_k = (tid & 1) << 2;
        int load_b_smem_k = tid >> 5;
        int load_b_smem_n = (tid & 31) << 2;

        if (n_block<N/BN && m_block<M/BM){
            for (int k_block=0;k_block<K/BK;++k_block){
                FLOAT4(shr_a[OFFSET(load_a_smem_m,load_a_smem_k,BK)]) = FLOAT4(a[OFFSET(m_block*BM+load_a_smem_m,BK*k_block+load_a_smem_k,K)]);
                FLOAT4(shr_b[OFFSET(load_b_smem_k,load_b_smem_n,BN)]) = FLOAT4(b[OFFSET(BK*k_block+load_b_smem_k,n_block*BN+load_b_smem_n,N)]);

                __syncthreads();

                for (int k=0;k<BK;++k){
                    for (int i=0;i<TM;++i){
                        for (int j=0;j<TN;++j){
                            reg_c[OFFSET(i,j,TN)] += shr_a[OFFSET(i+tid_y*TM,k,BK)]*shr_b[OFFSET(k,j+tid_x*TN,BN)];
                        }
                    }
                }

                __syncthreads();
            }
            
            for (int i=0;i<TM;++i){
                for (int j=0;j<TN;j+=4){
                    FLOAT4(c[OFFSET(m_block*BM+tid_y*TM+i,n_block*BN+tid_x*TN+j,N)])=FLOAT4(reg_c[OFFSET(i,j,TN)]);
                }
            }
        }


    }

__global__ void mySgemm_tiled_nonconf(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
        const int BM = 128;
        const int BN = 128;
        const int BK = 8;
        const int TM = 8;
        const int TN = 8;

        int n_block = blockIdx.x;
        int m_block = blockIdx.y;
        int tid = blockDim.x*threadIdx.y+threadIdx.x;
        int tid_x=threadIdx.x;
        int tid_y=threadIdx.y;

        __shared__ float shr_a[BK][BM];
        __shared__ float shr_b[BK][BN];
        float reg_c[TM][TN]={0.f};
        float r_load_a[4];
        // float r_load_b[4];
        float r_comp_a[TM];
        float r_comp_b[TN];

        int load_a_smem_m = tid >> 1;
        int load_a_smem_k = (tid & 1) << 2;
        int load_b_smem_k = tid >> 5;
        int load_b_smem_n = (tid & 31) << 2;

        if (n_block<N/BN && m_block<M/BM){
            for (int k_block=0;k_block<K/BK;++k_block){

                FLOAT4(r_load_a[0]) = FLOAT4(a[OFFSET(m_block*BM+load_a_smem_m,BK*k_block+load_a_smem_k,K)]);
                shr_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
                shr_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
                shr_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
                shr_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
                FLOAT4(shr_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[OFFSET(BK*k_block+load_b_smem_k,n_block*BN+load_b_smem_n,N)]);

                __syncthreads();

        #pragma unroll
                for (int k=0;k<BK;++k){
                    //use two non-adjacent lda128
                    FLOAT4(r_comp_a[0]) = FLOAT4(shr_a[k][tid_y * TM / 2         ]);
                    FLOAT4(r_comp_a[4]) = FLOAT4(shr_a[k][tid_y * TM / 2 + BM / 2]);
                    FLOAT4(r_comp_b[0]) = FLOAT4(shr_b[k][tid_x * TN / 2         ]);
                    FLOAT4(r_comp_b[4]) = FLOAT4(shr_b[k][tid_x * TN / 2 + BN / 2]);
        #pragma unroll
                    for (int i=0;i<TM;++i){
        #pragma unroll
                        for (int j=0;j<TN;++j){
                            reg_c[i][j] += r_comp_a[i] * r_comp_b[j];
                        }
                    }
                }

                __syncthreads();
            }

    #pragma unroll
            for (int i = 0; i < TM / 2; i++) {
                int store_c_gmem_m = m_block * BM + tid_y * TM / 2 + i;
                int store_c_gmem_n = n_block * BN + tid_x * TN / 2;
                int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
                FLOAT4(c[store_c_gmem_addr]) = FLOAT4(reg_c[i][0]);
                FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(reg_c[i][4]);
            }
    #pragma unroll
            for (int i = 0; i < TM / 2; i++) {
                int store_c_gmem_m = m_block * BM + BM / 2 + tid_y * TM / 2 + i;
                int store_c_gmem_n = n_block * BN + tid_x * TN / 2;
                int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
                FLOAT4(c[store_c_gmem_addr]) = FLOAT4(reg_c[i + TM / 2][0]);
                FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(reg_c[i + TM / 2][4]);
            }

        }
    }

__global__ void mySgemm_tiled_nonconf_doublebuffer(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
        const int BM = 128;
        const int BN = 128;
        const int BK = 8;
        const int TM = 8;
        const int TN = 8;

        int n_block = blockIdx.x;
        int m_block = blockIdx.y;
        int tid = blockDim.x*threadIdx.y+threadIdx.x;
        int tid_x=threadIdx.x;
        int tid_y=threadIdx.y;

        __shared__ float shr_a[2][BK][BM];
        __shared__ float shr_b[2][BK][BN];
        float reg_c[TM][TN]={0.f};
        float r_load_a[4];
        float r_load_b[4];
        float r_comp_a[TM];
        float r_comp_b[TN];

        int load_a_smem_m = tid >> 1;
        int load_a_smem_k = (tid & 1) << 2;
        int load_b_smem_k = tid >> 5;
        int load_b_smem_n = (tid & 31) << 2;

        if (n_block<N/BN && m_block<M/BM){

            FLOAT4(r_load_a[0]) = FLOAT4(a[OFFSET(m_block*BM+load_a_smem_m,load_a_smem_k,K)]);
            shr_a[0][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
            shr_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
            shr_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
            shr_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
            FLOAT4(shr_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(b[OFFSET(load_b_smem_k,n_block*BN+load_b_smem_n,N)]);


            __syncthreads();

            for (int k_block=1;k_block<K/BK;++k_block){
                short sign_load = k_block%2;
                short sign_comp = (k_block+1)%2;

                FLOAT4(r_load_a[0]) = FLOAT4(a[OFFSET(m_block*BM+load_a_smem_m,BK*k_block+load_a_smem_k,K)]);
                FLOAT4(r_load_b[0]) = FLOAT4(b[OFFSET(BK*k_block+load_b_smem_k,n_block*BN+load_b_smem_n,N)]);

                #pragma unroll
                for (int k=0;k<BK;++k){
                    //use two non-adjacent lda128
                    FLOAT4(r_comp_a[0]) = FLOAT4(shr_a[sign_comp][k][tid_y * TM / 2         ]);
                    FLOAT4(r_comp_a[4]) = FLOAT4(shr_a[sign_comp][k][tid_y * TM / 2 + BM / 2]);
                    FLOAT4(r_comp_b[0]) = FLOAT4(shr_b[sign_comp][k][tid_x * TN / 2         ]);
                    FLOAT4(r_comp_b[4]) = FLOAT4(shr_b[sign_comp][k][tid_x * TN / 2 + BN / 2]);
                    #pragma unroll
                    for (int i=0;i<TM;++i){
                    #pragma unroll
                        for (int j=0;j<TN;++j){
                            reg_c[i][j] += r_comp_a[i] * r_comp_b[j];
                        }
                    }
                }

                
                shr_a[sign_load][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
                shr_a[sign_load][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
                shr_a[sign_load][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
                shr_a[sign_load][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
                FLOAT4(shr_b[sign_load][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);


                __syncthreads();
            }
            #pragma unroll
            for (int k=0;k<BK;++k){
                //use two non-adjacent lda128
                FLOAT4(r_comp_a[0]) = FLOAT4(shr_a[1][k][tid_y * TM / 2         ]);
                FLOAT4(r_comp_a[4]) = FLOAT4(shr_a[1][k][tid_y * TM / 2 + BM / 2]);
                FLOAT4(r_comp_b[0]) = FLOAT4(shr_b[1][k][tid_x * TN / 2         ]);
                FLOAT4(r_comp_b[4]) = FLOAT4(shr_b[1][k][tid_x * TN / 2 + BN / 2]);
                #pragma unroll
                for (int i=0;i<TM;++i){
                    #pragma unroll
                    for (int j=0;j<TN;++j){
                        reg_c[i][j] += r_comp_a[i] * r_comp_b[j];
                    }
                }
            }
            

            #pragma unroll
            for (int i = 0; i < TM / 2; i++) {
                int store_c_gmem_m = m_block * BM + tid_y * TM / 2 + i;
                int store_c_gmem_n = n_block * BN + tid_x * TN / 2;
                int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
                FLOAT4(c[store_c_gmem_addr]) = FLOAT4(reg_c[i][0]);
                FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(reg_c[i][4]);
            }
            #pragma unroll
            for (int i = 0; i < TM / 2; i++) {
                int store_c_gmem_m = m_block * BM + BM / 2 + tid_y * TM / 2 + i;
                int store_c_gmem_n = n_block * BN + tid_x * TN / 2;
                int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
                FLOAT4(c[store_c_gmem_addr]) = FLOAT4(reg_c[i + TM / 2][0]);
                FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(reg_c[i + TM / 2][4]);
            }

        }
    }

float testMaxError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 0, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testCublasMaxError(const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);

    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error)
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

float testCublasPerformance(const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

int main() {

    const int M_list[13] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    const int N_list[13] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    const int K_list[13] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    const int outer_repeat = 10, inner_repeat = 1;

    {
        printf("\nKernal = cublas\n");

        {
            const int M = 512, N = 512, K = 512;
            float max_error = testCublasMaxError(M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {
            const int TESTNUM = 13;

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testCublasPerformance(M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernal = mySgemm_tiled\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            mySgemm_tiled;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {
            const int TESTNUM = 13;

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernal = mySgemm_tiled_nonconf\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            mySgemm_tiled_nonconf;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {
            const int TESTNUM = 13;

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernal = mySgemm_tiled_nonconf_doublebuffer\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            mySgemm_tiled_nonconf_doublebuffer;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {
            const int TESTNUM = 13;

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    return 0;
}
