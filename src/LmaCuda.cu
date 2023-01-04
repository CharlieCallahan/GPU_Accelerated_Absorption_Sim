#include "LmaCuda.cuh"


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
    cudaError_t cudaStatus = cudaMalloc((void **)&this->devicePtr, sizeof(float));
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error ",std::string(__FILE__), std::to_string(__LINE__));
    float zero = 0.0;
    cudaStatus = cudaMemcpy(this->devicePtr, (void*)&zero, sizeof(float), cudaMemcpyHostToDevice);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMemcpy Error ",std::string(__FILE__), std::to_string(__LINE__));
    
    int nBlocks = getNBlocks();
    gpuHelpers::kernel_vec_sum_sq<<<nBlocks, LM_CUDA_THREADS_PER_BLOCK>>>(this->devicePtr, devSum, this->length);
    cudaError_t cudaStatus = cudaDeviceSynchronize();
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

void gpuHelpers::kernel_vec_add(float* v0, float* v1, float* target, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<n){
        target[i] = v0[i] + v1[i];
    }
}

void gpuHelpers::kernel_vec_sub(float* v0, float* v1, float* target, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<n){
        target[i] = v0[i] - v1[i];
    }
}

void gpuHelpers::kernel_vec_sum_sq(float* v0, float* target, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<n){
        atomicAdd(target,v0[i]*v0[i]);
    }
}

void gpuHelpers::kernel_vec_add_inplace(float* v0, float* v1, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<n){
        atomicAdd(v0+i,v1[i]);
    }
}

CudaLMMatSparse::CudaLMMatSparse(int nnz, int nRows, int nCols){
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
    
}

CudaLMMatSparse::~CudaLMMatSparse(){
    cudaError_t cudaStatus;
    cudaStatus = cudaFree(this->d_csrValA);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaFree Error ",std::string(__FILE__), std::to_string(__LINE__));
    
    cudaStatus = cudaFree(this->d_csrRowPtrA);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaFree Error ",std::string(__FILE__), std::to_string(__LINE__));

    cudaStatus = cudaFree(this->d_csrColIndA);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaFree Error ",std::string(__FILE__), std::to_string(__LINE__));
}

LMMat* CudaLMMatSparse::operator*( LMMat* mat){
    CudaLMMatSparse* cmat = dynamic_cast<CudaLMMatSparse*>(mat);
    if(cmat == nullptr){
        std::cout << "Incorrect matrix type\n";
        return nullptr;
    }

    // cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE;
    // cusparseScsrsm2_analysis()

}

void gpuHelpers::kernel_ATApluslambdaI(
        float* d_csrValA,
        int* d_csrRowPtrA,
        int* d_csrColIndA,
        int nRowsA,
        float* C_cmajor,
        float lambda)
{
    //linear index
    int linear_ind = blockDim.x*blockIdx.x + threadIdx.x;
    if(linear_ind > (nRowsA*nRowsA-1)){
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
