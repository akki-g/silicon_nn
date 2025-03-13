#pragma once
#include "common/Device.h"

// Global device pointer (initialized to nullptr).
extern Device* globalDevice;

// Call this once to initialize the global device (if not already initialized).
void initializeGlobalDevice(DeviceType type);

void cleanupGlobalDevice();
