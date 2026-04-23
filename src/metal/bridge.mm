#include "bridge.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <assert.h>

struct ZLLM_Device {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
};

ZLLM_Device* zllm_metal_init(const char* shader_source) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return NULL;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            return NULL;
        }

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:shader_source]
                                                      options:nil
                                                        error:&error];
        if (library == nil) {
            NSLog(@"Metal shader compilation failed: %@", error);
            return NULL;
        }

        ZLLM_Device* z_device = (ZLLM_Device*)malloc(sizeof(ZLLM_Device));
        if (z_device == NULL) {
            return NULL;
        }

        z_device->device = device;
        z_device->queue = queue;
        z_device->library = library;
        return z_device;
    }
}

const char* zllm_metal_get_device_name(ZLLM_Device* device) {
    assert(device != NULL);
    return [[device->device name] UTF8String];
}

void* zllm_metal_create_buffer(ZLLM_Device* d, const void* data, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> buffer;
        if (data != NULL) {
            buffer = [d->device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
        } else {
            buffer = [d->device newBufferWithLength:size options:MTLResourceStorageModeShared];
        }
        return (__bridge_retained void*)buffer;
    }
}

void zllm_metal_release_buffer(void* buffer) {
    @autoreleasepool {
        // This will transfer ownership back to ARC and it will be released at the end of @autoreleasepool
        id<MTLBuffer> mtlBuffer = (__bridge_transfer id<MTLBuffer>)buffer;
        (void)mtlBuffer;
    }
}

void zllm_metal_dispatch(ZLLM_Device* d, const char* kernel_name, void** buffers, int n_buffers, int threads) {
    @autoreleasepool {
        id<MTLFunction> func = [d->library newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
        if (func == nil) {
            NSLog(@"Kernel function not found: %s", kernel_name);
            return;
        }

        NSError* error = nil;
        id<MTLComputePipelineState> state = [d->device newComputePipelineStateWithFunction:func error:&error];
        if (state == nil) {
            NSLog(@"Failed to create compute pipeline state: %@", error);
            return;
        }

        id<MTLCommandBuffer> cmdBuf = [d->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:state];

        for (int i = 0; i < n_buffers; i++) {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i];
            [encoder setBuffer:buffer offset:0 atIndex:i];
        }

        MTLSize gridSize = MTLSizeMake(threads, 1, 1);
        NSUInteger threadGroupSizeValue = state.maxTotalThreadsPerThreadgroup;
        if (threadGroupSizeValue > threads) {
            threadGroupSizeValue = threads;
        }
        MTLSize threadGroupSize = MTLSizeMake(threadGroupSizeValue, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
}

void zllm_metal_deinit(ZLLM_Device* device) {
    if (device == NULL) return;
    
    device->library = nil;
    device->queue = nil;
    device->device = nil;
    free(device);
}
