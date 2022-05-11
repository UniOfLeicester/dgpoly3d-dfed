#ifndef KERNEL_ASSEMBLEELEMS_H
#define KERNEL_ASSEMBLEELEMS_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "types.h"
#include <vector>


void elems_set_constant_mem(std::vector<Real>, std::vector<Real>, std::vector<int>);
extern "C" SYCL_EXTERNAL void
assembleElems(int, int, int, Real (*)[3], Real (*)[6], int (*)[4], int *, int *,
              Real *, int *, int *, sycl::nd_item<3> item_ct1,
              const sycl::stream &stream_ct1,
              dpct::accessor<Real, dpct::constant, 2> legendre,
              dpct::accessor<Real, dpct::constant, 2> nwElem,
              dpct::accessor<int, dpct::constant, 2> combinations);

#endif
