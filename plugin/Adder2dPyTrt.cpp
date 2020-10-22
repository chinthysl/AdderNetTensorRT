#include "Adder2dPlugin.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(adder2dpytrt, m) {
    py::class_<Adder2dPlugin>(m, "Adder2dPlugin")
    .def(py::init<const nvinfer1::Weights*, int , int, int, int, int>(),
     py::arg("weights"), py::arg("nbWeights"), py::arg("filterSize"), py::arg("nbfilter"), py::arg("stride"), py::arg("padding"))
    .def(py::init<const void *, size_t>(), py::arg("data"), py::arg("length"))
    .def("getSerializationSize", &Adder2dPlugin::getSerializationSize);
}
