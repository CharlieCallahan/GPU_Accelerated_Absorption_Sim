#include "LmaCuda.cuh"


CudaLMVec::CudaLMVec(float* data, int length){
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void **)&this->devicePtr, length * sizeof(float));
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error "+std::string(__FILE__)+": "+std::to_string(__LINE__));
    cudaStatus = cudaMemcpy(this->devicePtr, data, length * sizeof(float), cudaMemcpyHostToDevice);
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMemcpy Error "+std::string(__FILE__)+": "+std::to_string(__LINE__));
    this->length = length;
}

CudaLMVec::CudaLMVec(int length){
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void **)&this->devicePtr, length * sizeof(float));
	gpuHelpers::checkCudaErrors(cudaStatus, "cudaMalloc Error "+std::string(__FILE__)+": "+std::to_string(__LINE__));
    this->length = length;

}

LMVec* CudaLMVec::operator+(LMVec* vec){
    CudaLMVec* toAdd = dynamic_cast<CudaLMVec*> (vec);
    CudaLMVec* newVec = new CudaLMVec(length);
    int nBlocks = this->length/LM_CUDA_THREADS_PER_BLOCK + 1;

    gpuHelpers::kernel_vec_add<<<nBlocks,LM_CUDA_THREADS_PER_BLOCK>>>(this->devicePtr,toAdd->devicePtr,newVec->devicePtr,this->length);
}

void CudaLMVec::addInPlace( LMVec* vec){

}

/**
 * @brief Subtract vectors
 * 
 * @param vec 
 * @return LMVec 
 */
LMVec* CudaLMVec::operator-( LMVec* vec){

}

/**
 * @brief Return the sum of each value squared
 * 
 * @return float 
 */
float CudaLMVec::sumSq(){

} 

/**
 * @brief Make a copy of this vector
 * 
 * @return LMVec* 
 */
LMVec* CudaLMVec::copy(){

}

CudaLMVec::~CudaLMVec(){

}

void gpuHelpers::checkCudaErrors(cudaError_t &errorCode, std::string errorMessage)
{
    if (errorCode != cudaSuccess)
    {
        std::cout << errorMessage << ": " << cudaGetErrorString(errorCode) << "\n";
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
