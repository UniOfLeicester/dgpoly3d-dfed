#ifndef TYPES_H
#define TYPES_H

#define LEGENDRE_COLS 3
#define nwElem_shape0 27
#define combinations_shape0 10

    #ifdef USEFLOAT32
        typedef float Real;
    #else
        typedef double Real;
    #endif


struct Affine3
{
    Real B[3][3];
    Real C[3];
};

struct Vec3
{
    Real x;
    Real y;
    Real z;
};

struct Matrix3
{
    Real val[3][3];
};

struct MinBBox3
{
    Real x[2];
    Real y[2];
    Real z[2];
};

#endif
