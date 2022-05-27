#include <CL/sycl.hpp>
#include "kernel_assembleElems.h"
#include "types.h"
#include "affine.h"
#include "params.h"
#include "funcs.h"
#include "intel_atomic.hpp"
#include <stdio.h>
#include <vector>

// dpct::constant_memory<Real, 2> legendre(LEGENDRE_COLS, LEGENDRE_COLS);
// dpct::constant_memory<Real, 2> nwElem(nwElem_shape0, 4);
// dpct::constant_memory<int, 2> combinations(combinations_shape0, 3);

// void elems_set_constant_mem(std::vector<Real> legendreCoefs, std::vector<Real> nw_elem, std::vector<int> basisCombinations)
// {
//     dpct::device_ext &dev_ct1 = dpct::get_current_device();
//     sycl::queue &q_ct1 = dev_ct1.default_queue();
//     q_ct1
//         .memcpy(legendre.get_ptr(), legendreCoefs.data(),
//                 legendreCoefs.size() *
//                     sizeof(decltype(legendreCoefs)::value_type))
//         .wait();
//     q_ct1
//         .memcpy(nwElem.get_ptr(), nw_elem.data(),
//                 nw_elem.size() * sizeof(decltype(nw_elem)::value_type))
//         .wait();
//     q_ct1
//         .memcpy(combinations.get_ptr(), basisCombinations.data(),
//                 basisCombinations.size() *
//                     sizeof(decltype(basisCombinations)::value_type))
//         .wait();
// }

extern "C" SYCL_EXTERNAL void
assembleElems(int NT, int Nbasis, int Ngauss, Real nodes[][3],
              Real minBoundBox[][6], int tetrahedrons[][4],
              int *tetrahedrons2elem, int *NbasisCummulative, Real *Aval,
              int *Aindices, int *Aindptr, sycl::nd_item<3> item_ct1,
              const sycl::stream &stream_ct1,
              sycl::accessor<Real, 2, sycl::access::mode::read, sycl::access::target::constant_buffer> legendre,
              sycl::accessor<Real, 2, sycl::access::mode::read, sycl::access::target::constant_buffer> nwElem,
              sycl::accessor<int, 2, sycl::access::mode::read, sycl::access::target::constant_buffer> combinations)
{
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
              item_ct1.get_local_id(2);
    if (idx >= NT) return;

    int elementid = tetrahedrons2elem[idx];

    Vec3 node1, node2, node3, node4;

    node1.x = nodes[tetrahedrons[idx][0]][0];
    node1.y = nodes[tetrahedrons[idx][0]][1];
    node1.z = nodes[tetrahedrons[idx][0]][2];

    node2.x = nodes[tetrahedrons[idx][1]][0];
    node2.y = nodes[tetrahedrons[idx][1]][1];
    node2.z = nodes[tetrahedrons[idx][1]][2];

    node3.x = nodes[tetrahedrons[idx][2]][0];
    node3.y = nodes[tetrahedrons[idx][2]][1];
    node3.z = nodes[tetrahedrons[idx][2]][2];

    node4.x = nodes[tetrahedrons[idx][3]][0];
    node4.y = nodes[tetrahedrons[idx][3]][1];
    node4.z = nodes[tetrahedrons[idx][3]][2];

    Affine3 ref = genAffine3d(node1, node2, node3, node4);
    MinBBox3 mbb = genMinBoundingBox(minBoundBox, elementid);
    Real vol = volumetetr(node1, node2, node3, node4);

    int i, j, k, rowid, colid, pos;
    Matrix3 diff;
    Vec3 gradu, gradv, adv, quadrnode, h, m, h_0, h_1, snode, L0, L1;
    Real u, v, react, sumA;

    h.x = scaleh(mbb.x);
    m.x = scalem(mbb.x);

    h.y = scaleh(mbb.y);
    m.y = scalem(mbb.y);

    h.z = scaleh(mbb.z);
    m.z = scalem(mbb.z);

    h_0 = calcCoefh0(h);
    h_1 = calcCoefh1(h);

    for (i = 0; i < Nbasis; i++) {
        for (j = 0; j < Nbasis; j++) {
            sumA = 0.0f;

            for (k = 0; k < Ngauss; k++) {
                quadrnode.x = ref.B[0][0]*nwElem[k][0] + ref.B[0][1]*nwElem[k][1] + ref.B[0][2]*nwElem[k][2] + ref.C[0];
                quadrnode.y = ref.B[1][0]*nwElem[k][0] + ref.B[1][1]*nwElem[k][1] + ref.B[1][2]*nwElem[k][2] + ref.C[1];
                quadrnode.z = ref.B[2][0]*nwElem[k][0] + ref.B[2][1]*nwElem[k][1] + ref.B[2][2]*nwElem[k][2] + ref.C[2];

                snode.x = (quadrnode.x - m.x)/h.x;
                snode.y = (quadrnode.y - m.y)/h.y;
                snode.z = (quadrnode.z - m.z)/h.z;

                L0.x = eval1dLegendre0(legendre, combinations[i][0], snode.x);
                L0.y = eval1dLegendre0(legendre, combinations[i][1], snode.y);
                L0.z = eval1dLegendre0(legendre, combinations[i][2], snode.z);
                L1.x = eval1dLegendre1(legendre, combinations[i][0], snode.x);
                L1.y = eval1dLegendre1(legendre, combinations[i][1], snode.y);
                L1.z = eval1dLegendre1(legendre, combinations[i][2], snode.z);
                v = eval3dLegendre0(h_0, L0);
                gradv = eval3dLegendre1(h_0, h_1, L0, L1);

                L0.x = eval1dLegendre0(legendre, combinations[j][0], snode.x);
                L0.y = eval1dLegendre0(legendre, combinations[j][1], snode.y);
                L0.z = eval1dLegendre0(legendre, combinations[j][2], snode.z);
                L1.x = eval1dLegendre1(legendre, combinations[j][0], snode.x);
                L1.y = eval1dLegendre1(legendre, combinations[j][1], snode.y);
                L1.z = eval1dLegendre1(legendre, combinations[j][2], snode.z);
                u = eval3dLegendre0(h_0, L0);
                gradu = eval3dLegendre1(h_0, h_1, L0, L1);

                diff = diffusion(quadrnode);
                adv = advection(quadrnode);
                react = reaction(quadrnode);

                sumA += nwElem[k][3] * (dot(diff*gradu, gradv) + dot(adv, gradu)*v + react*u*v);
            }
            sumA *= vol;

            rowid = NbasisCummulative[elementid] + i;
            colid = NbasisCummulative[elementid] + j;
            pos = binarySearch(Aindices, Aindptr[rowid], Aindptr[rowid+1]-1, colid);

            // printf("element:%d - row:%d - col:%d - Aval:%4.2f\n",elementid, rowid, colid, sumA);

            if (pos == -1) {
                stream_ct1 << "Element integrals. Negative position.\n";
                break;
            }

            /*
            DPCT1039:0: The generated code assumes that "&Aval[pos]" points to
            the global memory address space. If it points to a local memory
            address space, replace "dpct::atomic_fetch_add" with
            "dpct::atomic_fetch_add<Real,
            sycl::access::address_space::local_space>".
            */
            intel_atomic::atomic_fetch_add(&Aval[pos], sumA);
        }
    }
}
