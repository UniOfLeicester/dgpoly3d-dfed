#ifndef FUNCS_H
#define FUNCS_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "types.h"

Vec3 operator+(Vec3 p1, Vec3 p2);
Vec3 operator-(Vec3 p1, Vec3 p2);
Vec3 operator*(Real s, Vec3 p1);
Vec3 operator/(Real s, Vec3 p1);
SYCL_EXTERNAL Real dot(Vec3 p1, Vec3 p2);
Vec3 cross(Vec3 p1, Vec3 p2);
SYCL_EXTERNAL Vec3 operator*(Matrix3 M, Vec3 v);
Real enorm(Vec3 p1);
Vec3 barycenter(Vec3 p1, Vec3 p2, Vec3 p3);
Vec3 createPerpendicularNode(Vec3 p1, Vec3 p2, Vec3 p3, Vec3 unitnormal);
Real volumetriangle(Vec3 p1, Vec3 p2, Vec3 p3);
SYCL_EXTERNAL Real volumetetr(Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4);
SYCL_EXTERNAL MinBBox3 genMinBoundingBox(Real minBoundBox[][6], int elementid);
SYCL_EXTERNAL Real scaleh(Real *mbb1d);
SYCL_EXTERNAL Real scalem(Real *mbb1d);
SYCL_EXTERNAL Vec3 calcCoefh0(Vec3 h);
SYCL_EXTERNAL Vec3 calcCoefh1(Vec3 h);
SYCL_EXTERNAL Real eval1dLegendre0(dpct::accessor<Real, dpct::constant, 2> legendre, int n,
                                   Real x);
SYCL_EXTERNAL Real eval1dLegendre1(dpct::accessor<Real, dpct::constant, 2> legendre, int n,
                                   Real x);
SYCL_EXTERNAL Real eval3dLegendre0(Vec3 h_0, Vec3 L0);
SYCL_EXTERNAL Vec3 eval3dLegendre1(Vec3 h_0, Vec3 h_1, Vec3 L0, Vec3 L1);
SYCL_EXTERNAL int binarySearch(int *arr, int l, int r, int x);

#endif
