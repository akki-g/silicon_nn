#!/bin/bash
set -e

OUTPUT_DIR="neural_network"
mkdir -p ${OUTPUT_DIR}

# List your source files.
SOURCES=("nn.cpp" "common/Device.cpp" "cpu/CPUDevice.cpp" "metal/MetalDevice.mm")
OBJS=()

# Compile each source file into an object file.
for src in "${SOURCES[@]}"; do
    obj="${src%.*}.o"
    OBJS+=("$obj")
    if [[ "$(uname)" == "Darwin" ]]; then
        clang++ -O3 -march=native -ffast-math -std=c++17 -arch arm64 -c "$src" -o "$obj"
    else
        g++ -O3 -march=native -ffast-math -std=c++17 -c "$src" -o "$obj"
    fi
done

# Link the object files into a shared library.
if [[ "$(uname)" == "Darwin" ]]; then
    clang++ -dynamiclib -arch arm64 -o ${OUTPUT_DIR}/libnn.dylib "${OBJS[@]}" -framework Metal -framework Foundation -Wl,-rpath,/usr/local/lib
else
    g++ -shared -fPIC -o ${OUTPUT_DIR}/libnn.so "${OBJS[@]}" -Wl,-rpath,/usr/local/lib
fi

echo "Build complete: shared library saved in ${OUTPUT_DIR}"
