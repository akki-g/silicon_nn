set -e 

if [[ "$(uname)" == "Darwin" ]]; then
    echo "Building for macOS"
    g++ -dynamiclib -std=c++11 -o libnn.dylib nn.cpp
else
    echo "Building for Linux"
    g++ -shared -fPIC -std=c++11 -o libnn.so nn.cpp
fi