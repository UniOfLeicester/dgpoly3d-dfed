#include <CL/sycl.hpp>
#include "affine.h"
#include "types.h"

SYCL_EXTERNAL Affine3 genAffine3d(Vec3 node1, Vec3 node2, Vec3 node3,
                                  Vec3 node4)
{
    const int N = 3;
    Real B[N][N] = {{0.5f*(node2.x - node1.x),   //0.5*(x2-x1)
                     0.5f*(node3.x - node1.x),   //0.5*(x3-x1)
                     0.5f*(node4.x - node1.x)},  //0.5*(x4-x1)
                    {0.5f*(node2.y - node1.y),   //0.5*(y2-y1)
                     0.5f*(node3.y - node1.y),   //0.5*(y3-y1)
                     0.5f*(node4.y - node1.y)},  //0.5*(y4-y1)
                    {0.5f*(node2.z - node1.z),   //0.5*(z2-z1)
                     0.5f*(node3.z - node1.z),   //0.5*(z3-z1)
                     0.5f*(node4.z - node1.z)}}; //0.5*(z4-z1)

    Real C[N] = {0.5f*(node2.x + node3.x + node4.x - node1.x),  //0.5*(x2+x3+x4-x1)
                 0.5f*(node2.y + node3.y + node4.y - node1.y),  //0.5*(y2+y3+y4-y1)
                 0.5f*(node2.z + node3.z + node4.z - node1.z)}; //0.5*(z2+z3+z4-z1)

    Affine3 temp;

    temp.B[0][0] = B[0][0];
    temp.B[0][1] = B[0][1];
    temp.B[0][2] = B[0][2];

    temp.B[1][0] = B[1][0];
    temp.B[1][1] = B[1][1];
    temp.B[1][2] = B[1][2];

    temp.B[2][0] = B[2][0];
    temp.B[2][1] = B[2][1];
    temp.B[2][2] = B[2][2];

    temp.C[0] = C[0];
    temp.C[1] = C[1];
    temp.C[2] = C[2];

    return temp;
}
