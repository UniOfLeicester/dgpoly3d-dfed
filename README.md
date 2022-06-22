# DPC++/SYCL

To compile the SYCL code (`sycl` directory):

```
make -f Makefile.[dpcpp|clang]
```

To run the code:

```
./main ../inout/e*_t**
```

where `*` and `**` are the number of elements (polyhedrons) and tetrahedrons respectively.

# License

Except where otherwise noted, this work is licensed under GNU GPLv3.

```
Copyright (C) 2022 Thomas Kappas

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
