# dGpoly3D

Code for the DiRAC Federation Project WP 3.2.1

Assembles the linear system of an interior penalty discontinuous Galerkin method for a second order equation with nonnegative characteristic form. This repository includes only the elemental kernel, both the native CUDA and the ported SYCL/DPC++ version.

More specifically, the problem that we assemble is $-Δu=f $ in $Ω $, $u=0 $ on $\vartheta Ω $ where $u(x,y,z) = \sin(\pi x) \sin(\pi y) \sin(\pi z) $. The meshes are cubes in $3D $ ($Ω=[0,1]^3 $) and the elements are polyhedral, stemming from the agglomeration of tetrahedrons. The polynomial degree was set to $p=2$ globally, the float precision to single and the python code from [this](https://github.com/TomK000/Dfed) private repository was used to create the input and output file (that can be found in the `inout` directory) for various mesh sizes, only the small of which were stored here using Git LFS.

# Compile & run

## Native CUDA code

To compile the native CUDA code (`cpp` directory):

```
make
```

To run the code:

```
./main ../inout/e*_t**
```

where `*` and `**` are the number of elements (polyhedrons) and tetrahedrons respectively.

*The native CUDA code was tested with GCC 9 and CUDA 11.*


## SYCL/DPC++ code

To compile the ported SYCL/DPC++ code (`sycl` directory):

```
make -f Makefile.[dpcpp|clang]
```

To run the code:

```
./main ../inout/e*_t**
```

where `*` and `**` are the number of elements (polyhedrons) and tetrahedrons respectively.

*The SYCL code was tested with Intel oneAPI 2022.*

# License

Except where otherwise noted, this work is licensed under GNU GPLv3.

```
Copyright (C) 2022 Thomas (Makis) Kappas

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
