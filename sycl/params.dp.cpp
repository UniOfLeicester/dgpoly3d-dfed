#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "params.h"
#include "types.h"
#include <math.h>

SYCL_EXTERNAL Matrix3 diffusion(Vec3 node)
{
    Matrix3 diff;

    diff.val[0][0] = 1.0f;
    diff.val[0][1] = 0.0f;
    diff.val[0][2] = 0.0f;

    diff.val[1][0] = 0.0f;
    diff.val[1][1] = 1.0f;
    diff.val[1][2] = 0.0f;

    diff.val[2][0] = 0.0f;
    diff.val[2][1] = 0.0f;
    diff.val[2][2] = 1.0f;

    return diff;
}

SYCL_EXTERNAL Vec3 advection(Vec3 node)
{
    Vec3 adv;

    adv.x = 0.0f;
    adv.y = 0.0f;
    adv.z = 0.0f;

    return adv;
}

SYCL_EXTERNAL Real reaction(Vec3 node)
{
    Real react;

    react = 0.0f;

    return react;
}
