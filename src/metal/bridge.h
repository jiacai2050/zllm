#ifndef ZLLM_METAL_BRIDGE_H
#define ZLLM_METAL_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZLLM_Device ZLLM_Device;

/**
 * Initialize a Metal device.
 * Returns a pointer to a ZLLM_Device on success, or NULL on failure.
 */
ZLLM_Device* zllm_metal_init(void);

/**
 * Get the name of the Metal device.
 * The returned string is owned by the device and remains valid until zllm_metal_deinit is called.
 */
const char* zllm_metal_get_device_name(ZLLM_Device* device);

/**
 * Deinitialize the Metal device and free associated resources.
 */
void zllm_metal_deinit(ZLLM_Device* device);

#ifdef __cplusplus
}
#endif

#endif // ZLLM_METAL_BRIDGE_H
