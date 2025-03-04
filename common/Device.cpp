#include "Device.h"
#include "../cpu/CPUDevice.h"
#include "../metal/MetalDevice.h"

Device* Device::create(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return new CPUDevice();
        case DeviceType::GPU:
            return new MetalDevice();
        default:
            return nullptr;
    }
}