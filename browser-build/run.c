// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"

extern const iree_hal_executable_library_header_t **
jit_energy_linked_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t *environment);
typedef struct {
  iree_host_size_t shape_rank;
  const iree_hal_dim_t *shape;
  iree_hal_element_type_t element_type;
  const void *data;
  iree_host_size_t data_length;
} input_buffer_descriptor_t;

// Helper function to allocate a buffer view and push it as an input to a
// runtime call.
// This function performs the following operations:
// 1. Allocates a device-local buffer view with the provided data
// 2. Pushes the buffer view to the runtime call's input list
// 3. Releases the buffer view reference (the call retains its own reference)
static iree_status_t push_back_buffer_view(iree_hal_device_t *device,
                                           input_buffer_descriptor_t input,
                                           iree_runtime_call_t *call) {
  iree_hal_buffer_view_t *buffer_view = NULL;
  iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), input.shape_rank, input.shape,
      input.element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(input.data, input.data_length), &buffer_view);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(call, buffer_view);
  }
  iree_hal_buffer_view_release(buffer_view);
  return status;
}

// A function to create the HAL device from the different backend targets.
// The HAL device is returned based on the implementation, and it must be
// released by the caller.
iree_status_t create_device_with_static_loader(iree_allocator_t host_allocator,
                                               iree_hal_device_t **out_device) {
  // Set parameters for the device created in the next step.
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  // Register the statically linked executable library.
  const iree_hal_executable_library_query_fn_t libraries[] = {
      jit_energy_linked_library_query,
  };
  iree_hal_executable_loader_t *library_loader = NULL;
  iree_status_t status = iree_hal_static_library_loader_create(
      IREE_ARRAYSIZE(libraries), libraries,
      iree_hal_executable_import_provider_null(), host_allocator,
      &library_loader);

  // Use the default host allocator for buffer allocations.
  iree_string_view_t identifier = iree_make_cstring_view("local-sync");
  iree_hal_allocator_t *device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(identifier, host_allocator,
                                            host_allocator, &device_allocator);
  }

  // Create a shared proactor pool for async I/O. The device retains the pool
  // so we release our reference immediately after device creation.
  iree_async_proactor_pool_t *proactor_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), host_allocator,
        &proactor_pool);
  }

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  // Create the device and release the executor and loader afterwards.
  if (iree_status_is_ok(status)) {
    status = iree_hal_sync_device_create(
        identifier, &params, &create_params, /*loader_count=*/1,
        &library_loader, device_allocator, host_allocator, out_device);
  }
  iree_async_proactor_pool_release(proactor_pool);

  iree_hal_allocator_release(device_allocator);
  iree_hal_executable_loader_release(library_loader);
  return status;
}

static iree_runtime_instance_t *instance;
static iree_hal_device_t *device;
static iree_runtime_session_t *session;
static int checked(iree_status_t st) {
  if (iree_status_is_ok(st))
    return 0;
  iree_status_fprint(stderr, st);
  iree_status_free(st);
  return 1;
}
int setup(unsigned char *data, int size) {
  iree_runtime_instance_options_t opt;
  iree_runtime_instance_options_initialize(&opt);
  if (checked(iree_runtime_instance_create(&opt, iree_allocator_system(),
                                           &instance)))
    return 1;
  if (checked(
          create_device_with_static_loader(iree_allocator_system(), &device)))
    return 2;
  iree_runtime_session_options_t opts;
  iree_runtime_session_options_initialize(&opts);
  if (checked(iree_runtime_session_create_with_device(
          instance, &opts, device, iree_allocator_system(), &session)))
    return 3;
  return checked(iree_runtime_session_append_bytecode_module_from_memory(
      session, iree_make_const_byte_span(data, size), iree_allocator_null()));
}
int evaluate(int n, int s, int l, float *r, int *z, float *q, int *sr, int *lr,
             float *energy, float *gradient) {
  iree_runtime_call_t call;
  if (checked(iree_runtime_call_initialize_by_name(
          session, iree_make_cstring_view("jit_energy.main"), &call)))
    return 1;
  iree_hal_dim_t dims[5][2] = {{n, 3}, {n}, {0}, {2, s}, {2, l}};
  input_buffer_descriptor_t inp[] = {
      {2, dims[0], IREE_HAL_ELEMENT_TYPE_FLOAT_32, r, n * 12},
      {1, dims[1], IREE_HAL_ELEMENT_TYPE_SINT_32, z, n * 4},
      {0, NULL, IREE_HAL_ELEMENT_TYPE_FLOAT_32, q, 4},
      {2, dims[3], IREE_HAL_ELEMENT_TYPE_SINT_32, sr, s * 8},
      {2, dims[4], IREE_HAL_ELEMENT_TYPE_SINT_32, lr, l * 8}};
  int err = 0;
  for (int i = 0; i < 5 && !err; i++)
    err = checked(push_back_buffer_view(device, inp[i], &call));
  if (!err)
    err = checked(iree_runtime_call_invoke(&call, 0));
  for (int i = 0; i < 2 && !err; i++) {
    iree_hal_buffer_view_t *v = NULL;
    err = checked(iree_runtime_call_outputs_pop_front_buffer_view(&call, &v));
    if (!err)
      err = checked(iree_hal_device_transfer_d2h(
          device, iree_hal_buffer_view_buffer(v), 0, i ? gradient : energy,
          i ? n * 12 : 4, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
          iree_infinite_timeout()));
    iree_hal_buffer_view_release(v);
  }
  iree_runtime_call_deinitialize(&call);
  return err;
}
