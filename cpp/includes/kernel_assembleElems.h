#ifndef KERNEL_ASSEMBLEELEMS_H
#define KERNEL_ASSEMBLEELEMS_H

#include "types.h"
#include <vector>


void elems_set_constant_mem(std::vector<Real>, std::vector<Real>, std::vector<int>);
extern "C" __global__ void assembleElems(int, int, int, Real (*)[3], Real (*)[6], int (*)[4], int *, int *, Real *, int *, int *);

#endif
