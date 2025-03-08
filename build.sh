#!/bin/bash
set -e

OUTPUT_DIR="neural_network"
mkdir -p ${OUTPUT_DIR}

# List your source files.
SOURCES=("src/nn.cpp" "src/common/Device.cpp" "src/cpu/CPUDevice.cpp" "src/metal/MetalDevice.mm")
OBJS=()

for src in "${SOURCES[@]}"; do
    obj="${src%.*}.o"
    OBJS+=("$obj")
    if [[ "$(uname)" == "Darwin" ]]; then
        clang++ -O3 -march=native -ffast-math -std=c++17 -arch arm64 -I./include -c "$src" -o "$obj"
    else
        g++ -O3 -march=native -ffast-math -std=c++17 -I./include -c "$src" -o "$obj"
    fi
done

if [[ "$(uname)" == "Darwin" ]]; then
    clang++ -dynamiclib -arch arm64 -I./include -o ${OUTPUT_DIR}/libnn.dylib "${OBJS[@]}" -framework Metal -framework Foundation -Wl,-rpath,/usr/local/lib
else
    g++ -shared -fPIC -I./include -o ${OUTPUT_DIR}/libnn.so "${OBJS[@]}" -Wl,-rpath,/usr/local/lib
fi

echo "Build complete: shared library saved in ${OUTPUT_DIR}"