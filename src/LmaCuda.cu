#include "LmaCuda.cuh"
#include "assert.h"
#include "cusolverSp.h"

CuSparseContext::CuSparseContext(){
    cudaError_t cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	gpuHelpers::checkCudaErrors(cudaStatus, "Device syc error",std::string(__FILE__), std::to_string(__LINE__));
    this->status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    status = cusparseSetStream(handle, stream);
    assert(CUSPARSE_STATUS_SUCCESS == status);
}

CudaLMVec::CudaLMVec(float* data, int length){
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void **)&this->devicePtr, length * sizeof(float));
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));
    cudaStatus = cudaMemcpy(this->devicePtr, data, length * sizeof(float), cudaMemcpyHostToDevice);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMemcpy Error ",std::string(__FILE__), std::to_string(__LINE__));
    this->length = length;
}

CudaLMVec::CudaLMVec(int length){
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void **)&this->devicePtr, length * sizeof(float));
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));
    this->length = length;

}

LMVec* CudaLMVec::operator+(LMVec* vec){
    CudaLMVec* toAdd = dynamic_cast<CudaLMVec*> (vec);
    CudaLMVec* newVec = new CudaLMVec(length);
    int nBlocks = getNBlocks();
    gpuHelpers::kernel_vec_add<<<nBlocks,LM_CUDA_THREADS_PER_BLOCK>>>(this->devicePtr,toAdd->devicePtr,newVec->devicePtr,this->length);
    cudaError_t cudaStatus = cudaDeviceSynchronize();
	gpuHelpers::checkCudaErrors(cudaStatus, "Device syc error",std::string(__FILE__), std::to_string(__LINE__));
    return newVec;
}

void CudaLMVec::addInPlace( LMVec* vec){
    CudaLMVec* toAdd = dynamic_cast<CudaLMVec*> (vec);
    int nBlocks = getNBlocks();
    gpuHelpers::kernel_vec_add_inplace<<<nBlocks,LM_CUDA_THREADS_PER_BLOCK>>>(this->devicePtr,toAdd->devicePtr,this->length);
    cudaError_t cudaStatus = cudaDeviceSynchronize();
	gpuHelpers::checkCudaErrors(cudaStatus, "Device syc error",std::string(__FILE__), std::to_string(__LINE__));
}

/**
 * @brief Subtract vectors
 * 
 * @param vec 
 * @return LMVec 
 */
LMVec* CudaLMVec::operator-( LMVec* vec){
    CudaLMVec* toAdd = dynamic_cast<CudaLMVec*> (vec);
    CudaLMVec* newVec = new CudaLMVec(length);
    int nBlocks = getNBlocks();
    gpuHelpers::kernel_vec_sub<<<nBlocks,LM_CUDA_THREADS_PER_BLOCK>>>(this->devicePtr,toAdd->devicePtr,newVec->devicePtr,this->length);
    cudaError_t cudaStatus = cudaDeviceSynchronize();
	gpuHelpers::checkCudaErrors(cudaStatus, "Device syc error",std::string(__FILE__), std::to_string(__LINE__));
    return newVec;
}

/**
 * @brief Return the sum of each value squared
 * 
 * @return float 
 */
float CudaLMVec::sumSq(){
    //create target
    float* devSum;
    cudaError_t cudaStatus = cudaMalloc((void **)&devSum, sizeof(float));
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));
    float zero = 0.0;
    cudaStatus = cudaMemcpy(devSum, (void*)&zero, sizeof(float), cudaMemcpyHostToDevice);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMemcpy Error ",std::string(__FILE__), std::to_string(__LINE__));
    
    int nBlocks = getNBlocks();
    gpuHelpers::kernel_vec_sum_sq<<<nBlocks, LM_CUDA_THREADS_PER_BLOCK>>>(this->devicePtr, devSum, this->length);
    cudaStatus = cudaDeviceSynchronize();
    float result;
    cudaStatus = cudaMemcpy((void*)&result, (void*)devSum, sizeof(float), cudaMemcpyDeviceToHost);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMemcpy Error ",std::string(__FILE__), std::to_string(__LINE__));

    cudaStatus = cudaFree(devSum);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaFree Error ",std::string(__FILE__), std::to_string(__LINE__));

    return result;
} 

/**
 * @brief Make a copy of this vector
 * 
 * @return LMVec* 
 */
LMVec* CudaLMVec::copy(){
    CudaLMVec* newVec = new CudaLMVec(this->length);
    cudaError_t cudaStatus = cudaMemcpy((void*)newVec->devicePtr, (void*)this->devicePtr, this->length*sizeof(float), cudaMemcpyDeviceToDevice);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMemcpy Error ",std::string(__FILE__), std::to_string(__LINE__));
    return newVec;
}

CudaLMVec::~CudaLMVec(){
    cudaError_t cudaStatus = cudaFree(this->devicePtr);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaFree Error ",std::string(__FILE__), std::to_string(__LINE__));
}

int CudaLMVec::getNBlocks(){
    return this->length/LM_CUDA_THREADS_PER_BLOCK + 1;
}

void gpuHelpers::checkCudaErrors(cudaError_t &errorCode, std::string errorMessage, std::string srcFile, std::string srcLine)
{
    if (errorCode != cudaSuccess)
    {
        std::cout <<srcFile<<":"<<srcLine<<": "<< errorMessage << ": " << cudaGetErrorString(errorCode) << "\n";
    }
}

__global__ void gpuHelpers::kernel_vec_add(float* v0, float* v1, float* target, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<n){
        target[i] = v0[i] + v1[i];
    }
}

__global__ void gpuHelpers::kernel_vec_sub(float* v0, float* v1, float* target, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<n){
        target[i] = v0[i] - v1[i];
    }
}

__global__ void gpuHelpers::kernel_vec_sum_sq(float* v0, float* target, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<n){
        atomicAdd(target,v0[i]*v0[i]);
    }
}

__global__ void gpuHelpers::kernel_vec_add_inplace(float* v0, float* v1, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<n){
        atomicAdd(v0+i,v1[i]);
    }
}

CudaLMMatSparse::CudaLMMatSparse(int nnz, int nRows, int nCols){
    if(!context){
        context = new CuSparseContext();
    }
    this->nnz = nnz;
    this->nRows = nRows;
    this->nCols = nCols;
    
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void **)&this->d_csrValA, nnz * sizeof(float));
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));
    cudaStatus = cudaMalloc((void **)&this->d_csrRowPtrA, (nRows+1) * sizeof(int));
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));
    cudaStatus = cudaMalloc((void **)&this->d_csrColIndA, nnz * sizeof(int));
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));
    cusparseStatus_t status = cusparseCreateMatDescr(&desc);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL );

}

CudaLMMatSparse::CudaLMMatSparse(int nRows){
        cusparseStatus_t status = cusparseCreateMatDescr(&desc);
        assert(CUSPARSE_STATUS_SUCCESS == status);
        cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL );
        cudaError_t cudaStatus = cudaMalloc((void **)&this->d_csrRowPtrA, (nRows+1) * sizeof(int));
	    gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));
    
    }

CudaLMMatSparse::~CudaLMMatSparse(){
    cudaError_t cudaStatus;
    cudaStatus = cudaFree(this->d_csrValA);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaFree Error ",std::string(__FILE__), std::to_string(__LINE__));
    
    cudaStatus = cudaFree(this->d_csrRowPtrA);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaFree Error ",std::string(__FILE__), std::to_string(__LINE__));

    cudaStatus = cudaFree(this->d_csrColIndA);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaFree Error ",std::string(__FILE__), std::to_string(__LINE__));

    cusparseDestroyMatDescr(desc);
}

LMMat* CudaLMMatSparse::calcMTMpLambdaI(float lambda){
    float* d_C_Cmajor_dense; //device dense matrix result

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void **)&d_C_Cmajor_dense, this->nRows*this->nRows * sizeof(float));
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));

    int nBlocks = (this->nRows*this->nRows)/LM_CUDA_THREADS_PER_BLOCK + 1;

    gpuHelpers::kernel_ATApluslambdaI<<<nBlocks, LM_CUDA_THREADS_PER_BLOCK>>>(this->d_csrValA,this->d_csrRowPtrA,this->d_csrColIndA,this->nRows,d_C_Cmajor_dense,lambda);
    
    CudaLMMatSparse* output = new CudaLMMatSparse(nRows);

    gpuHelpers::cuSparseConvertDenseToCSR(this->context, output, d_C_Cmajor_dense, nRows, nRows);
    cudaFree(d_C_Cmajor_dense);
    return output;
}

LMVec* CudaLMMatSparse::operator*( LMVec* vec){
    CudaLMVec* in_vec = dynamic_cast<CudaLMVec*>(vec);
    CudaLMVec* out_vec = new CudaLMVec(in_vec->length);
    int nBlocks = in_vec->length;
    gpuHelpers::sparseMatVecMult<<<nBlocks,LM_CUDA_THREADS_PER_BLOCK>>>(d_csrValA,d_csrRowPtrA,d_csrColIndA,in_vec->length,in_vec->devicePtr,out_vec->devicePtr);
    return out_vec;
}

LMVec* CudaLMMatSparse::solve(LMVec* b){
    CudaLMVec* in_vec = dynamic_cast<CudaLMVec*>(b);
    CudaLMVec* out_vec = new CudaLMVec(in_vec->length);

    cusolverSpHandle_t solverHandle;
    cusolverStatus_t solverStatus = cusolverSpCreate(&solverHandle);
    assert(solverStatus == CUSOLVER_STATUS_SUCCESS);
    solverStatus = cusolverSpSetStream(solverHandle, context->stream);
    assert(solverStatus == CUSOLVER_STATUS_SUCCESS);

    int* d_singularity;
    cudaError_t status = cudaMalloc((void**)&d_singularity,sizeof(int));
    gpuHelpers::checkCudaErrors(status, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));

    cusolverSpScsrlsvqr(
        solverHandle,
        in_vec->length,
        this->nnz,
        this->desc,
        this->d_csrValA,
        this->d_csrRowPtrA,
        this->d_csrColIndA,
        in_vec->devicePtr,
        0.01,
        0,
        out_vec->devicePtr,
        d_singularity
    );

    cudaDeviceSynchronize();
    gpuHelpers::checkCudaErrors(status, "cudaDeviceSynchronize Error ",std::string(__FILE__), std::to_string(__LINE__));
    cudaFree(d_singularity);
    cusolverSpDestroy(solverHandle);
    return out_vec;
}

__global__ void gpuHelpers::kernel_ATApluslambdaI(
        float* d_csrValA,
        int* d_csrRowPtrA,
        int* d_csrColIndA,
        int nRowsA,
        float* C_cmajor,
        float lambda)
{
    //linear index
    int linear_ind = blockDim.x*blockIdx.x + threadIdx.x;
    if(linear_ind > (nRowsA*nRowsA-1)){ //out of bounds
        return;
    }

    //convert to mat index
    int i = linear_ind/nRowsA;
    int j = linear_ind%nRowsA;

    int startInd_i = d_csrRowPtrA[i];
    int startInd_j = d_csrRowPtrA[j];

    int endInd_i = d_csrRowPtrA[i+1];
    int endInd_j = d_csrRowPtrA[j+1];

    int di = endInd_i - startInd_i;
    int dj = endInd_j - startInd_j;
    int delta;
    int* smallerRow;
    int* largerRow;
    int lStartInd; //large start index
    int sStartInd; //smaller start index
    int largeMax; //max column index for larger row
    if(di < dj){
        delta = di;
        smallerRow = d_csrColIndA + startInd_i;
        largerRow = d_csrColIndA + startInd_j;
        sStartInd = startInd_i;
        lStartInd = startInd_j;
        largeMax = dj;
    } else {
        delta = dj;
        smallerRow = d_csrColIndA + startInd_i;
        largerRow = d_csrColIndA + startInd_j;
        sStartInd = startInd_j;
        lStartInd = startInd_i;
        largeMax = di;
    }

    float sum = 0.0;
    if(i==j){
        sum = lambda; //add lambda*I
    }

    //this assumes that the csr column indices are sorted!
    int lOffset = 0; //offset into largerRow
    for(int k = 0; k < delta; k++){
        int sInd = smallerRow[k]; //currentSmallInd
        int lInd = largerRow[lOffset];
        if(lInd == sInd){
            sum+=(d_csrValA[lStartInd + lOffset]*d_csrValA[sStartInd + k]);
        } else {
            while(lInd < sInd && (lOffset < largeMax)){
                lOffset++;
                lInd = largerRow[lOffset];
            }
            if(lInd == sInd){
                sum+=(d_csrValA[lStartInd + lOffset]*d_csrValA[sStartInd + k]);
            }
        }
    }
    C_cmajor[linear_ind] = sum;
}

__global__ void gpuHelpers::sparseMatVecMult(
        float* d_csrValA,
        int* d_csrRowPtrA,
        int* d_csrColIndA,
        int nRowsA,
        float* vec_in,
        float* vec_out
    )
{
    int linear_ind = blockDim.x*blockIdx.x + threadIdx.x;
    if(linear_ind > (nRowsA-1)){
        return;
    }

    int startLoc = d_csrRowPtrA[linear_ind];
    int endLoc = d_csrRowPtrA[linear_ind+1];

    vec_out[linear_ind] = 0.0;
    for (int i = startLoc; i < endLoc; i++){
        vec_out[linear_ind]+=vec_in[d_csrColIndA[i]]*d_csrValA[i];
    }
}

void gpuHelpers::cuSparseConvertDenseToCSR(CuSparseContext* ctx, CudaLMMatSparse* emptyTarget, float* d_denseMat, int nCols, int nRows){
    float threshold = 0.00001;
    size_t lworkInBytes;
    cusparseStatus_t status = cusparseSpruneDense2csr_bufferSizeExt(
    ctx->handle,
    nRows,
    nCols,
    d_denseMat,
    nRows,
    &threshold,
    emptyTarget->desc,
    emptyTarget->d_csrValA,
    emptyTarget->d_csrRowPtrA,
    emptyTarget->d_csrColIndA,
    &lworkInBytes);

    char* d_work; //work memory
    cudaError_t cudaStatus = cudaMalloc((void**)&d_work, lworkInBytes);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));
    assert(CUSPARSE_STATUS_SUCCESS == status);

/* compute csrRowPtrC and nnzC */
    int nnzC;
    status = cusparseSpruneDense2csrNnz(
        ctx->handle,
        nRows,
        nCols,
        d_denseMat,
        nRows,
        &threshold,
        emptyTarget->desc,
        emptyTarget->d_csrRowPtrA,
        &nnzC, /* host */
        d_work);
    
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStatus = cudaDeviceSynchronize();
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaDeviceSynchronize Error ",std::string(__FILE__), std::to_string(__LINE__));
    
    /* compute csrColIndC and csrValC */
    cudaStatus = cudaMalloc ((void**)&emptyTarget->d_csrColIndA, sizeof(int  ) * nnzC );
    cudaStatus = cudaMalloc ((void**)&emptyTarget->d_csrValA   , sizeof(float) * nnzC );
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));
    
    status = cusparseSpruneDense2csr(
        ctx->handle,
        nRows,
        nCols,
        d_denseMat,
        nRows,
        &threshold,
        emptyTarget->desc,
        emptyTarget->d_csrValA,
        emptyTarget->d_csrRowPtrA,
        emptyTarget->d_csrColIndA,
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStatus = cudaDeviceSynchronize();
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaDeviceSynchronize Error ",std::string(__FILE__), std::to_string(__LINE__));
    
    cudaStatus = cudaFree(d_work);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaFree Error ",std::string(__FILE__), std::to_string(__LINE__));

    //done
}