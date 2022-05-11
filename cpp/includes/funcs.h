#ifndef FUNCS_H
#define FUNCS_H

#include "types.h"


__device__ Vec3 operator+(Vec3 p1, Vec3 p2);
__device__ Vec3 operator-(Vec3 p1, Vec3 p2);
__device__ Vec3 operator*(Real s, Vec3 p1);
__device__ Vec3 operator/(Real s, Vec3 p1);
__device__ Real dot(Vec3 p1, Vec3 p2);
__device__ Vec3 cross(Vec3 p1, Vec3 p2);
__device__ Vec3 operator*(Matrix3 M, Vec3 v);
__device__ Real enorm(Vec3 p1);
__device__ Vec3 barycenter(Vec3 p1, Vec3 p2, Vec3 p3);
__device__ Vec3 createPerpendicularNode(Vec3 p1, Vec3 p2, Vec3 p3, Vec3 unitnormal);
__device__ Real volumetriangle(Vec3 p1, Vec3 p2, Vec3 p3);
__device__ Real volumetetr(Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4);
__device__ MinBBox3 genMinBoundingBox(Real minBoundBox[][6], int elementid);
__device__ Real scaleh(Real *mbb1d);
__device__ Real scalem(Real *mbb1d);
__device__ Vec3 calcCoefh0(Vec3 h);
__device__ Vec3 calcCoefh1(Vec3 h);
__device__ Real eval1dLegendre0(Real legendre[][LEGENDRE_COLS], int n, Real x);
__device__ Real eval1dLegendre1(Real legendre[][LEGENDRE_COLS], int n, Real x);
__device__ Real eval3dLegendre0(Vec3 h_0, Vec3 L0);
__device__ Vec3 eval3dLegendre1(Vec3 h_0, Vec3 h_1, Vec3 L0, Vec3 L1);
__device__ int binarySearch(int *arr, int l, int r, int x);

#endif
