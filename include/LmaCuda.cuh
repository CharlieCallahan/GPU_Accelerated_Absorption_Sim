/* Copyright (c) 2022 Charlie Callahan
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef LMA_CUDA_CUH
#define LMA_CUDA_CUH
#define LM_CUDA_THREADS_PER_BLOCK 128
#include "Lma.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <iostream>
#include "cusparse.h"

/**
 * @brief Cusparse context singleton object
 * 
 */
class CuSparseContext{
    public:
    /**
     * @brief Initialize cusparse context.
     * 
     */
    CuSparseContext();

    ~CuSparseContext(){
        if (handle        ) cusparseDestroy(handle);
        if (stream        ) cudaStreamDestroy(stream);
    }
    cusparseHandle_t handle;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    cudaStream_t stream;
};

/**
 * @brief LMA Matrices and vectors implemented in CUDA
 * 
 */

class CudaLMVec : public LMVec{
    
    public:
    /**
     * @brief Allocate on GPU and copy from CPU
     * 
     * @param data 
     * @param length 
     */
    CudaLMVec(float* data, int length);

    /**
     * @brief Empty vector
     * 
     * @param length 
     */
    CudaLMVec(int length);
    
    LMVec* operator+( LMVec* vec) override;

    void addInPlace( LMVec* vec) override;

    /**
     * @brief Subtract vectors
     * 
     * @param vec 
     * @return LMVec 
     */
    LMVec* operator-( LMVec* vec) override;

    /**
     * @brief Return the sum of each value squared
     * 
     * @return float 
     */
    float sumSq() override; 

    /**
     * @brief Make a copy of this vector
     * 
     * @return LMVec* 
     */
    LMVec* copy() override;

    ~CudaLMVec();

    float* devicePtr;
    int length;

    private:
    int getNBlocks();
};

class CudaLMMatSparse : public LMMat{
    public:
    /**
     * @brief Construct a new Cuda LM Mat Sparse object
     * 
     * @param nnz number of nonzero elements 
     */
    CudaLMMatSparse(int nnz, int nRows, int nCols);
    
    CudaLMMatSparse(int nRows);

    ~CudaLMMatSparse();

    LMMat* calcMTMpLambdaI(float lambda) override;

    // /**
    //  * @brief Transpose
    //  * 
    //  * @return LMMat 
    //  */
    // LMMat* transpose() override;

    /**
     * @brief solves the linear equation A*x=b for x - (A is this matrix)
     * 
     * @param b 
     * @return LMVec* x
     */
    LMVec* solve(LMVec* b) override;

    /**
     * @brief Matrix vector multiplication
     * 
     */
    LMVec* operator*( LMVec* vec) override;

    static CuSparseContext* context;

    int nnz;
    int nRows;
    int nCols;
    float* d_csrValA;
    int* d_csrRowPtrA;
    int* d_csrColIndA;
    cusparseMatDescr_t desc;

};

CuSparseContext* CudaLMMatSparse::context = new CuSparseContext();

/**
 * @brief Helper functions
 * 
 */
namespace gpuHelpers
{
    void cuSparseConvertDenseToCSR(CuSparseContext* ctx, CudaLMMatSparse* emptyTarget, float* d_denseMat, int nCols, int nRows);

    void checkCudaErrors(cudaError_t &errorCode, std::string errorMessage, std::string srcFile, std::string srcLine);

    /**
     * @brief Add these two vectors together, put result in target
     * 
     * @param v0 
     * @param v1 
     * @param target
     * @return __global__ 
     */
    __global__ void kernel_vec_add(float* v0, float* v1, float* target, int n);

    /**
     * @brief Adds v1 to v0 in place
     * 
     * @param v0 
     * @param v1 
     * @return __global__ 
     */
    __global__ void kernel_vec_add_inplace(float* v0, float* v1, int n);

    __global__ void kernel_vec_sub(float* v0, float* v1, float* target, int n);

    /**
     * @brief sum the squares of each element, uses atomic addition
     * 
     * @param v0 vector
     * @param target single float value to hold sum
     * @return __global__ 
     */
    __global__ void kernel_vec_sum_sq(float* v0, float* target, int n);

    /**
     * @brief C = transform(A)*A + lambda*I with A in sparse csr format and C in dense column major format
     * 
     * 
     */
    __global__ void kernel_ATApluslambdaI(
        float* d_csrValA,
        int* d_csrRowPtrA,
        int* d_csrColIndA,
        int nRowsA,
        float* C_cmajor,
        float lambda);

    /**
     * @brief vec_out = A*vec_in, should be parallelized row wise : 1 row per thread
     * 
     * @param d_csrValA 
     * @param d_csrRowPtrA 
     * @param d_csrColIndA 
     * @param nRowsA 
     * @param vec_in 
     * @param vec_out 
     * @return __global__ 
     */
    __global__ void sparseMatVecMult(
        float* d_csrValA,
        int* d_csrRowPtrA,
        int* d_csrColIndA,
        int nRowsA,
        float* vec_in,
        float* vec_out
    );


} // namespace gpuHelpers


#endif