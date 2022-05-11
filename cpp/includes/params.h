#ifndef PARAMS_H
#define PARAMS_H

#include "types.h"


__device__ Matrix3 diffusion(Vec3 node);
__device__ Vec3 advection(Vec3 node);
__device__ Real reaction(Vec3 node);

#endif
