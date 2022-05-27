#ifndef KERNEL_ASSEMBLEELEMS_H
#define KERNEL_ASSEMBLEELEMS_H

#include <CL/sycl.hpp>
#include "types.h"
#include <vector>


void elems_set_constant_mem(std::vector<Real>, std::vector<Real>, std::vector<int>);
extern "C" SYCL_EXTERNAL void
assembleElems(int, int, int, Real (*)[3], Real (*)[6], int (*)[4], int *, int *,
              Real *, int *, int *, sycl::nd_item<3> item_ct1,
              const sycl::stream &stream_ct1,
              sycl::accessor<Real, 2, sycl::access::mode::read, sycl::access::target::constant_buffer> legendre,
              sycl::accessor<Real, 2, sycl::access::mode::read, sycl::access::target::constant_buffer> nwElem,
              sycl::accessor<int, 2, sycl::access::mode::read, sycl::access::target::constant_buffer> combinations);

#endif
