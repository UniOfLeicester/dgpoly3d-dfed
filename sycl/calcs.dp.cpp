#include <CL/sycl.hpp>
#include "calcs.h"
#include "types.h"
#include "kernel_assembleElems.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <cmath>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(int code, const char *file, int line, bool abort = true)
{
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
    // dpct::device_ext &dev_ct1 = dpct::get_current_device();
    // sycl::queue &q_ct1 = dev_ct1.default_queue();

    // auto platformlist = sycl::platform::get_platforms();
    // std::cout << "List of detected devices:" << "\n";
    // for (auto p : platformlist) {
    //     auto devicelist = p.get_devices(sycl::info::device_type::all);
    //     for(auto d : devicelist) {
    //         std::string device_vendor = d.get_info<sycl::info::device::vendor>();
    //         std::cout<<d.get_info<sycl::info::device::name>()<<"\n";
    //     }
    // }

    sycl::queue q_ct1{ sycl::default_selector{} };
    std::cout << "Running on: " << q_ct1.get_device().get_info<sycl::info::device::name>() << "\n";

    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

    // sycl::default_selector device_selector;
    // sycl::queue queue(device_selector);
    // std::cout << "Running on "
    //     << queue.get_device().get_info<sycl::info::device::name>()
    //     << "\n";

    /*
    DPCT1026:2: The call to cudaEventCreate was removed because this call is
    redundant in DPC++.
    */
    /*
    DPCT1026:3: The call to cudaEventCreate was removed because this call is
    redundant in DPC++.
    */
    float milliseconds;

    Real (*d_nodes)[3];
    d_nodes = (Real(*)[3])sycl::malloc_device(
        nodes.size() * sizeof(decltype(nodes)::value_type), q_ct1);
    q_ct1
        .memcpy(d_nodes, nodes.data(),
                nodes.size() * sizeof(decltype(nodes)::value_type))
        .wait();

    Real (*d_mbb)[6];
    d_mbb = (Real(*)[6])sycl::malloc_device(
        mbb.size() * sizeof(decltype(mbb)::value_type), q_ct1);
    q_ct1
        .memcpy(d_mbb, mbb.data(),
                mbb.size() * sizeof(decltype(mbb)::value_type))
        .wait();

    int (*d_tetrahedrons)[4];
    d_tetrahedrons = (int(*)[4])sycl::malloc_device(
        tetrahedrons.size() * sizeof(decltype(tetrahedrons)::value_type),
        q_ct1);
    q_ct1
        .memcpy(d_tetrahedrons, tetrahedrons.data(),
                tetrahedrons.size() *
                    sizeof(decltype(tetrahedrons)::value_type))
        .wait();

    int *d_tetrahedrons2elem;
    d_tetrahedrons2elem = (int *)sycl::malloc_device(
        tetrahedrons2elem.size() *
            sizeof(decltype(tetrahedrons2elem)::value_type),
        q_ct1);
    q_ct1
        .memcpy(d_tetrahedrons2elem, tetrahedrons2elem.data(),
                tetrahedrons2elem.size() *
                    sizeof(decltype(tetrahedrons2elem)::value_type))
        .wait();

    int *d_NbasisCummulative;
    d_NbasisCummulative = (int *)sycl::malloc_device(
        NbasisCummulative.size() *
            sizeof(decltype(NbasisCummulative)::value_type),
        q_ct1);
    q_ct1
        .memcpy(d_NbasisCummulative, NbasisCummulative.data(),
                NbasisCummulative.size() *
                    sizeof(decltype(NbasisCummulative)::value_type))
        .wait();

    Real *d_Aval;
    d_Aval = (Real *)sycl::malloc_device(
        Aval.size() * sizeof(decltype(Aval)::value_type), q_ct1);
    q_ct1
        .memcpy(d_Aval, Aval.data(),
                Aval.size() * sizeof(decltype(Aval)::value_type))
        .wait();

    int *d_Aindices;
    d_Aindices = (int *)sycl::malloc_device(
        Aindices.size() * sizeof(decltype(Aindices)::value_type), q_ct1);
    q_ct1
        .memcpy(d_Aindices, Aindices.data(),
                Aindices.size() * sizeof(decltype(Aindices)::value_type))
        .wait();

    int *d_Aindptr;
    d_Aindptr = (int *)sycl::malloc_device(
        Aindptr.size() * sizeof(decltype(Aindptr)::value_type), q_ct1);
    q_ct1
        .memcpy(d_Aindptr, Aindptr.data(),
                Aindptr.size() * sizeof(decltype(Aindptr)::value_type))
        .wait();

    // Set the constant memory
    // elems_set_constant_mem(legendreCoefs, nw_elem, basisCombinations);

    // Buffers for the 3 arrays in constant memory
    sycl::buffer<Real, 2> legendreCoefs_buf(legendreCoefs.data(), sycl::range<2>{LEGENDRE_COLS, LEGENDRE_COLS});
    sycl::buffer<Real, 2> nw_elem_buf(nw_elem.data(), sycl::range<2>{nwElem_shape0, 4});
    sycl::buffer<int, 2> basisCombinations_buf(basisCombinations.data(), sycl::range<2>{combinations_shape0, 3});

    // Launch the kernel
    int gridSize, blockSize;
    blockSize = 128;
    gridSize = ceil(NT / (float) blockSize);
    // printf("%d - %d\n", gridSize, blockSize);

    /*
    DPCT1012:4: Detected kernel execution time measurement pattern and generated
    an initial code for time measurements in SYCL. You can change the way time
    is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
    /*
    DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    stop = q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // extern dpct::constant_memory<Real, 2> legendre;
        // extern dpct::constant_memory<Real, 2> nwElem;
        // extern dpct::constant_memory<int, 2> combinations;

        // legendre.init();
        // nwElem.init();
        // combinations.init();

        // auto legendre_acc_ct1 = legendre.get_access(cgh);
        // auto nwElem_acc_ct1 = nwElem.get_access(cgh);
        // auto combinations_acc_ct1 = combinations.get_access(cgh);

        auto legendre_acc_ct1 = legendreCoefs_buf.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
        auto nwElem_acc_ct1 = nw_elem_buf.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
        auto combinations_acc_ct1 = basisCombinations_buf.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize) *
                                  sycl::range<3>(1, 1, blockSize),
                              sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item_ct1) {
                assembleElems(NT, Nbasis, Ngauss3, d_nodes, d_mbb,
                              d_tetrahedrons, d_tetrahedrons2elem,
                              d_NbasisCummulative, d_Aval, d_Aindices,
                              d_Aindptr, item_ct1, stream_ct1, legendre_acc_ct1,
                              nwElem_acc_ct1, combinations_acc_ct1);
            });
    });
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());
    /*
    DPCT1012:5: Detected kernel execution time measurement pattern and generated
    an initial code for time measurements in SYCL. You can change the way time
    is measured depending on your goals.
    */
    stop.wait();
    stop_ct1 = std::chrono::steady_clock::now();
    milliseconds =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    printf("Time for kernel assembleElems (ms): %.2f\n", milliseconds);



    // Fetch bach the values of A
    std::vector<Real> Aval_elems(Aval.size());
    q_ct1
        .memcpy(Aval_elems.data(), d_Aval,
                Aval.size() * sizeof(decltype(Aval)::value_type))
        .wait();

    // Free device memory
    sycl::free(d_nodes, q_ct1);
    sycl::free(d_mbb, q_ct1);
    sycl::free(d_tetrahedrons, q_ct1);
    sycl::free(d_tetrahedrons2elem, q_ct1);
    sycl::free(d_NbasisCummulative, q_ct1);
    sycl::free(d_Aval, q_ct1);
    sycl::free(d_Aindices, q_ct1);
    sycl::free(d_Aindptr, q_ct1);

    /*
    DPCT1026:7: The call to cudaEventDestroy was removed because this call is
    redundant in DPC++.
    */
    /*
    DPCT1026:8: The call to cudaEventDestroy was removed because this call is
    redundant in DPC++.
    */

    return Aval_elems;

}
