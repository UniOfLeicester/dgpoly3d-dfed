#ifndef AFFINE_H
#define AFFINE_H

#include <CL/sycl.hpp>
#include "types.h"

SYCL_EXTERNAL Affine3 genAffine3d(Vec3 node1, Vec3 node2, Vec3 node3,
                                  Vec3 node4);

#endif
