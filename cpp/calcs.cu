#include "calcs.h"
#include "types.h"
#include "kernel_assembleElems.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime_api.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


std::vector<Real> calcs(int NT,
                        int Nbasis,
                        int Ngauss3,
                        std::vector<Real> nodes,
                        std::vector<Real> mbb,
                        std::vector<int> tetrahedrons,
                        std::vector<int> tetrahedrons2elem,
                        std::vector<int> NbasisCummulative,
                        std::vector<Real> Aval,
                        std::vector<int> Aindices,
                        std::vector<int> Aindptr,
                        std::vector<Real> legendreCoefs,
                        std::vector<Real> nw_elem,
                        std::vector<int> basisCombinations)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    Real (*d_nodes)[3];
    cudaMalloc(&d_nodes, nodes.size() * sizeof(decltype(nodes)::value_type));
    cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(decltype(nodes)::value_type), cudaMemcpyHostToDevice);

    Real (*d_mbb)[6];
    cudaMalloc(&d_mbb, mbb.size() * sizeof(decltype(mbb)::value_type));
    cudaMemcpy(d_mbb, mbb.data(), mbb.size() * sizeof(decltype(mbb)::value_type), cudaMemcpyHostToDevice);

    int (*d_tetrahedrons)[4];
    cudaMalloc(&d_tetrahedrons, tetrahedrons.size() * sizeof(decltype(tetrahedrons)::value_type));
    cudaMemcpy(d_tetrahedrons, tetrahedrons.data(), tetrahedrons.size() * sizeof(decltype(tetrahedrons)::value_type), cudaMemcpyHostToDevice);

    int *d_tetrahedrons2elem;
    cudaMalloc(&d_tetrahedrons2elem, tetrahedrons2elem.size() * sizeof(decltype(tetrahedrons2elem)::value_type));
    cudaMemcpy(d_tetrahedrons2elem, tetrahedrons2elem.data(), tetrahedrons2elem.size() * sizeof(decltype(tetrahedrons2elem)::value_type), cudaMemcpyHostToDevice);

    int *d_NbasisCummulative;
    cudaMalloc(&d_NbasisCummulative, NbasisCummulative.size() * sizeof(decltype(NbasisCummulative)::value_type));
    cudaMemcpy(d_NbasisCummulative, NbasisCummulative.data(), NbasisCummulative.size() * sizeof(decltype(NbasisCummulative)::value_type), cudaMemcpyHostToDevice);

    Real *d_Aval;
    cudaMalloc(&d_Aval, Aval.size() * sizeof(decltype(Aval)::value_type));
    cudaMemcpy(d_Aval, Aval.data(), Aval.size() * sizeof(decltype(Aval)::value_type), cudaMemcpyHostToDevice);

    int *d_Aindices;
    cudaMalloc(&d_Aindices, Aindices.size() * sizeof(decltype(Aindices)::value_type));
    cudaMemcpy(d_Aindices, Aindices.data(), Aindices.size() * sizeof(decltype(Aindices)::value_type), cudaMemcpyHostToDevice);

    int *d_Aindptr;
    cudaMalloc(&d_Aindptr, Aindptr.size() * sizeof(decltype(Aindptr)::value_type));
    cudaMemcpy(d_Aindptr, Aindptr.data(), Aindptr.size() * sizeof(decltype(Aindptr)::value_type), cudaMemcpyHostToDevice);


    // Set the constant memory
    elems_set_constant_mem(legendreCoefs, nw_elem, basisCombinations);

    // Launch the kernel
    int gridSize, blockSize;
    blockSize = 128;
    gridSize = ceil(NT / (float) blockSize);
    // printf("%d - %d\n", gridSize, blockSize);

    cudaEventRecord(start);
    assembleElems<<<gridSize, blockSize>>>(NT, Nbasis, Ngauss3,
                                           d_nodes, d_mbb, d_tetrahedrons, d_tetrahedrons2elem,
                                           d_NbasisCummulative, d_Aval, d_Aindices, d_Aindptr);
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel assembleElems (ms): %.2f\n", milliseconds);



    // Fetch bach the values of A
    std::vector<Real> Aval_elems(Aval.size());
    cudaMemcpy(Aval_elems.data(), d_Aval, Aval.size() * sizeof(decltype(Aval)::value_type), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_nodes);
    cudaFree(d_mbb);
    cudaFree(d_tetrahedrons);
    cudaFree(d_tetrahedrons2elem);
    cudaFree(d_NbasisCummulative);
    cudaFree(d_Aval);
    cudaFree(d_Aindices);
    cudaFree(d_Aindptr);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return Aval_elems;

}
