#ifndef AFFINE_H
#define AFFINE_H

#include "types.h"


__device__ Affine3 genAffine3d(Vec3 node1, Vec3 node2, Vec3 node3, Vec3 node4);

#endif
