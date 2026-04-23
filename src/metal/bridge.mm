#include "bridge.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <assert.h>

struct ZLLM_Device {
    id<MTLDevice> device;
};

ZLLM_Device* zllm_metal_init(void) {
    // We use @autoreleasepool to ensure any temporary objects are cleaned up,
    // although MTLCreateSystemDefaultDevice doesn't typically create many.
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return NULL;
        }

        ZLLM_Device* z_device = (ZLLM_Device*)malloc(sizeof(ZLLM_Device));
        if (z_device == NULL) {
            return NULL;
        }

        z_device->device = device;
        return z_device;
    }
}

const char* zllm_metal_get_device_name(ZLLM_Device* device) {
    assert(device != NULL);
    // name property returns an NSString. UTF8String returns a const char* 
    // that is valid as long as the NSString is alive.
    // Since the device owns the MTLDevice, and the name is likely a property of it,
    // this should be safe until deinit.
    return [[device->device name] UTF8String];
}

void zllm_metal_deinit(ZLLM_Device* device) {
    if (device == NULL) return;
    
    // In ARC, setting the id to nil will decrement the reference count.
    device->device = nil;
    free(device);
}
