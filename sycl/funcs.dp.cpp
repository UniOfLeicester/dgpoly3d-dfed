#include <CL/sycl.hpp>
#include "funcs.h"
#include "types.h"
#include "params.h"
#include <math.h>

namespace sycl = cl::sycl;


Vec3 operator+(Vec3 p1, Vec3 p2)
{
    //p1+p2
    Vec3 result;
    result.x = p1.x + p2.x;
    result.y = p1.y + p2.y;
    result.z = p1.z + p2.z;
    return result;
}

Vec3 operator-(Vec3 p1, Vec3 p2)
{
    //p1-p2
    Vec3 result;
    result.x = p1.x - p2.x;
    result.y = p1.y - p2.y;
    result.z = p1.z - p2.z;
    return result;
}

Vec3 operator*(Real s, Vec3 p1)
{
    //scalar multiplication
    Vec3 result;
    result.x = s*p1.x;
    result.y = s*p1.y;
    result.z = s*p1.z;
    return result;
}

Vec3 operator/(Real s, Vec3 p1)
{
    //scalar division
    Vec3 result;
    result.x = p1.x/s;
    result.y = p1.y/s;
    result.z = p1.z/s;
    return result;
}

Real dot(Vec3 p1, Vec3 p2)
{
    //p1*p2 (dot product)
    Real result;
    result = p1.x*p2.x + p1.y*p2.y + p1.z*p2.z;
    return result;
}

Vec3 cross(Vec3 p1, Vec3 p2)
{
    //p1xp2 (cross product)
    Vec3 result;
    result.x = p1.y*p2.z - p1.z*p2.y;
    result.y = p1.z*p2.x - p1.x*p2.z;
    result.z = p1.x*p2.y - p1.y*p2.x;
    return result;
}

Vec3 operator*(Matrix3 M, Vec3 v)
{
    //M*v (matrix-vector product)
    Vec3 result;
    result.x = M.val[0][0]*v.x + M.val[0][1]*v.y + M.val[0][2]*v.z;
    result.y = M.val[1][0]*v.x + M.val[1][1]*v.y + M.val[1][2]*v.z;
    result.z = M.val[2][0]*v.x + M.val[2][1]*v.y + M.val[2][2]*v.z;
    return result;
}

Real enorm(Vec3 p1)
{
    //Euclidean norm
    Real result;
    result = sycl::sqrt((float)(dot(p1, p1)));
    return result;
}

Vec3 barycenter(Vec3 p1, Vec3 p2, Vec3 p3)
{
    //Barycenter of a triangle in 3D
    Vec3 result;
    result.x = (p1.x + p2.x + p3.x)/3.0f;
    result.y = (p1.y + p2.y + p3.y)/3.0f;
    result.z = (p1.z + p2.z + p3.z)/3.0f;
    return result;
}

Vec3 createPerpendicularNode(Vec3 p1, Vec3 p2, Vec3 p3, Vec3 unitnormal)
{
    //Extra node, perpendicular to the unit normal vector (for face)
    Vec3 result;
    Vec3 barcenter = barycenter(p1, p2, p3);
    Real length = enorm(p2-p1);
    result.x = barcenter.x + length*unitnormal.x;
    result.y = barcenter.y + length*unitnormal.y;
    result.z = barcenter.z + length*unitnormal.z;
    return result;
}

Real volumetriangle(Vec3 p1, Vec3 p2, Vec3 p3)
{
    // 1/2 * |(p2-p1)x(p3-p1)|
    Real vol;
    vol = 0.5f * enorm(cross(p2-p1, p3-p1));
    return vol;
}

Real volumetetr(Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4)
{
    // 1/6 * |(p4-p1) * ((p2-p1)x(p3-p1))|
    Real vol;
    vol = sycl::fabs((float)(dot(p4 - p1, cross(p2 - p1, p3 - p1)))) / 6.0f;
    return vol;
}

MinBBox3 genMinBoundingBox(Real minBoundBox[][6], int elementid)
{
    // For polyhydral meshes (agglomerated tetrahedrons).
    MinBBox3 result;

    //x
    result.x[0] = minBoundBox[elementid][0];
    result.x[1] = minBoundBox[elementid][1];
    //y
    result.y[0] = minBoundBox[elementid][2];
    result.y[1] = minBoundBox[elementid][3];
    //z
    result.z[0] = minBoundBox[elementid][4];
    result.z[1] = minBoundBox[elementid][5];

    return result;
}

Real scaleh(Real *mbb1d)
{
    Real h = (mbb1d[1] - mbb1d[0])/2.0;
    return h;
}

Real scalem(Real *mbb1d)
{
    Real m = (mbb1d[0] + mbb1d[1])/2.0;
    return m;
}

Vec3 calcCoefh0(Vec3 h)
{
    Vec3 result;
    result.x = sycl::sqrt(1.0 / h.x);
    result.y = sycl::sqrt(1.0 / h.y);
    result.z = sycl::sqrt(1.0 / h.z);

    return result;
}

Vec3 calcCoefh1(Vec3 h)
{
    Vec3 result;
    result.x = sycl::sqrt(1.0 / (h.x * h.x * h.x));
    result.y = sycl::sqrt(1.0 / (h.y * h.y * h.y));
    result.z = sycl::sqrt(1.0 / (h.z * h.z * h.z));

    return result;
}

Real eval1dLegendre0(sycl::accessor<Real, 2, sycl::access::mode::read, sycl::access::target::constant_buffer> legendre, int n,
                                   Real x)
{
    Real L = legendre[n][0];
    for (int i = 1; i <= n; i++)
        L = L*x + legendre[n][i];

    return L;
}

Real eval1dLegendre1(sycl::accessor<Real, 2, sycl::access::mode::read, sycl::access::target::constant_buffer> legendre, int n,
                                   Real x)
{
    Real L = legendre[n][0] * n;
    for (int i = 1; i < n; i++)
        L = (L*x + (legendre[n][i]*(n-i)));

    return L;
}

Real eval3dLegendre0(Vec3 h_0, Vec3 L0)
{
    Real result;
    result = h_0.x*L0.x * h_0.y*L0.y * h_0.z*L0.z;

    return result;
}

Vec3 eval3dLegendre1(Vec3 h_0, Vec3 h_1, Vec3 L0, Vec3 L1)
{
    Vec3 result;

    result.x = h_1.x*L1.x * h_0.y*L0.y * h_0.z*L0.z;
    result.y = h_0.x*L0.x * h_1.y*L1.y * h_0.z*L0.z;
    result.z = h_0.x*L0.x * h_0.y*L0.y * h_1.z*L1.z;

    return result;
}

int binarySearch(int *arr, int l, int r, int x)
{
    // A iterative binary search function. It returns
    // location of x in given array arr[l..r] if present,
    // otherwise -1
    while (l <= r)
    {
        int m = l + (r-l)/2;

        // Check if x is present at mid
        if (arr[m] == x)
            return m;

        // If x greater, ignore left half
        if (arr[m] < x)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }

    // if we reach here, then element was
    // not present
    return -1;
}
