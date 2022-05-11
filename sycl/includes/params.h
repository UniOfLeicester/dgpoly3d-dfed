#ifndef PARAMS_H
#define PARAMS_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "types.h"

SYCL_EXTERNAL Matrix3 diffusion(Vec3 node);
SYCL_EXTERNAL Vec3 advection(Vec3 node);
SYCL_EXTERNAL Real reaction(Vec3 node);

#endif
