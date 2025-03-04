#!/bin/bash
set -e

OUTPUT_DIR="neural_network"

if [[ "$(uname)" == "Darwin" ]]; then
    echo "Building for macOS"
    g++ -O3 -march=native -ffast-math -dynamiclib -std=c++11 -o ${OUTPUT_DIR}/libnn.dylib nn.cpp -Wl,-rpath,/usr/local/lib
else
    echo "Building for Linux"
    g++ -O3 -march=native -ffast-math -shared -fPIC -std=c++11 -o ${OUTPUT_DIR}/libnn.so nn.cpp -Wl,-rpath,/usr/local/lib
fi
echo "Build complete: shared library saved in ${OUTPUT_DIR}"
