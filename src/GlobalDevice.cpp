#include "GlobalDevice.h"
#include "cpu/CPUDevice.h"
#include "metal/MetalDevice.h"

Device* globalDevice = nullptr;

void initializeGlobalDevice(DeviceType type) {
    if (!globalDevice) {
        globalDevice = Device::create(type);
    }
}

void cleanupGlobalDevice() {
    if (globalDevice) {
        globalDevice->cleanup();
        delete globalDevice;
        globalDevice = nullptr;
    }
}