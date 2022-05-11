#include "types.h"
#include "calcs.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>


template<typename T>
std::vector<T> readBinaryFile(std::string fileName)
{
    // Create an input file stream and open the file
    std::ifstream fin;
    fin.open(fileName, std::ios::binary);
    // Check if the file has opened succesfully
    if (fin.fail()) {
        std::cerr << "Error - Failed to open file " << fileName << std::endl;
        exit(-1);
    }

    // Get file size
    fin.seekg(0, fin.end);
    size_t length = fin.tellg();
    fin.seekg(0, fin.beg);

    // Read file and store it in vector
    std::vector<T> ret(length / sizeof(T));
    fin.read((char *)ret.data(), length);

    // Close the file
    fin.close();

    return ret;
}

template<typename T>
bool isClose(T x, T y, T rel_tol=0.01, T abs_tol=0.00001)
{
    // Same as Python's math.isclose(a, b, rel_tol, abs_tol)
    // return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    bool res;

    if (std::abs(x - y) <= std::max(rel_tol * std::max(std::abs(x), std::abs(y)), abs_tol)) {
        res = true;
    }
    else {
        res = false;
    }

    return res;
}

int main(int argc, char *argv[])
{
    std::string dirName = argv[1];
    if (dirName.back() != '/') {
        dirName += "/";
    }

    int NT;
    std::vector<int> NT_vec;
    NT_vec = readBinaryFile<int>(dirName + "NT");
    NT = NT_vec[0];

    int Nbasis;
    std::vector<int> Nbasis_vec;
    Nbasis_vec = readBinaryFile<int>(dirName + "Nbasis");
    Nbasis = Nbasis_vec[0];

    int Ngauss3;
    std::vector<int> Ngauss3_vec;
    Ngauss3_vec = readBinaryFile<int>(dirName + "Ngauss3");
    Ngauss3 = Ngauss3_vec[0];

    std::vector<Real> nodes;
    nodes = readBinaryFile<Real>(dirName + "nodes");

    std::vector<Real> mbb;
    mbb = readBinaryFile<Real>(dirName + "mbb");

    std::vector<int> tetrahedrons;
    tetrahedrons = readBinaryFile<int>(dirName + "tetrahedrons");

    std::vector<int> tetrahedrons2elem;
    tetrahedrons2elem = readBinaryFile<int>(dirName + "tetrahedrons2elem");

    std::vector<int> NbasisCummulative;
    NbasisCummulative = readBinaryFile<int>(dirName + "NbasisCummulative");

    std::vector<Real> Aval;
    Aval = readBinaryFile<Real>(dirName + "Adata");

    std::vector<Real> Aval_elems_ref;
    Aval_elems_ref = readBinaryFile<Real>(dirName + "Adata_elems");

    std::vector<int> Aindices;
    Aindices = readBinaryFile<int>(dirName + "Aindices");

    std::vector<int> Aindptr;
    Aindptr = readBinaryFile<int>(dirName + "Aindptr");

    std::vector<Real> legendreCoefs;
    legendreCoefs = readBinaryFile<Real>(dirName + "legendreCoefs");

    std::vector<Real> nw_elem;
    nw_elem = readBinaryFile<Real>(dirName + "nw_elem");

    std::vector<int> basisCombinations;
    basisCombinations = readBinaryFile<int>(dirName + "basisCombinations");

    std::vector<Real> Aval_elems;

    // for(const auto& value: nw_elem) {
    //     std::cout << value << std::endl;
    // }

    int Nelements = *std::max_element(tetrahedrons2elem.begin(), tetrahedrons2elem.end()) + 1;
    std::cout << "p: 2 (hardcoded)" << std::endl; // hardcoded 2nd order polynomial degree
    std::cout << "Number of elements (polyhedrons): " << Nelements << std::endl;
    std::cout << "Number of simplices (tetrahedrons): " << NT << std::endl;
    std::cout << "N (degrees of freedom): " << Aindptr.size() - 1 << std::endl;
    std::cout << "nnz (total): " << Aval.size() << std::endl;
    std::cout << "nnz that elements kernel will populate: " << Nelements * Nbasis << std::endl;
    std::cout << "% of nnz that elements kernel will populate: " << 100.0 * (Nelements * Nbasis) / Aval.size() << std::endl;
    std::cout << std::endl;

    Aval_elems = calcs(NT, Nbasis, Ngauss3, nodes, mbb, tetrahedrons, tetrahedrons2elem,
                       NbasisCummulative, Aval, Aindices, Aindptr,
                       legendreCoefs, nw_elem, basisCombinations);


    // Compare with the reference values
    bool valuesCorrect = true;
    int NwrongValues = 0;
    for (long unsigned int i = 0; i < Aval_elems_ref.size(); i++) {
        if (!isClose<Real>(Aval_elems_ref[i], Aval_elems[i])) {
            // std::cout << "Wrong value: " << Aval_elems_ref[i] << " =/= " << Aval_elems[i] << std::endl;
            valuesCorrect = false;
            NwrongValues += 1;
            // break;
        }
        // else {
        //     std::cout << "Correct value: " << Aval_elems_ref[i] << " = " << Aval_elems[i] << std::endl;
        // }
    }

    if (valuesCorrect)
        std::cout << "All values are correct" << std::endl;
    else
        std::cout << (100.0 * NwrongValues / Aval_elems_ref.size()) << "% of values are wrong" << std::endl;


    return 0;
}
