#!/bin/bash
set -e

OUTPUT_DIR="neural_network"

if [[ "$(uname)" == "Darwin" ]]; then
    echo "Building for macOS"
    clang++ -O3 -march=native -ffast-math -dynamiclib -std=c++11 -fopenmp -arch arm64 \
        nn.cpp common/Device.cpp cpu/CPUDevice.cpp metal/MetalDevice.mm -fno-objc-arc \
        -framework Metal -framework Foundation \
        -o neural_network/libnn.dylib -Wl,-rpath,/usr/local/lib
else
    echo "Building for Linux"
    g++ -O3 -march=native -ffast-math -shared -fPIC -std=c++11 \
        nn.cpp common/Device.cpp cpu/CPUDevice.cpp metal/MetalDevice.mm \
        -o ${OUTPUT_DIR}/libnn.so -Wl,-rpath,/usr/local/lib
fi

echo "Build complete: shared library saved in ${OUTPUT_DIR}"
