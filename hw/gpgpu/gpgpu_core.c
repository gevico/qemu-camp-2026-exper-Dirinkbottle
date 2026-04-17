/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "fpu/softfloat-helpers.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    uint32_t i;

    num_threads = MIN(num_threads, (uint32_t)GPGPU_WARP_SIZE);
    memset(warp, 0, sizeof(*warp));

    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    memcpy(warp->block_id, block_id, sizeof(warp->block_id));
    warp->active_mask = (num_threads == GPGPU_WARP_SIZE) ?
                        UINT32_MAX : ((1u << num_threads) - 1);

    for (i = 0; i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *lane = &warp->lanes[i];

        lane->pc = pc;
        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
        lane->active = i < num_threads;
        lane->fcsr = 0;

        memset(&lane->fp_status, 0, sizeof(lane->fp_status));
        set_float_rounding_mode(float_round_nearest_even, &lane->fp_status);
        set_float_exception_flags(0, &lane->fp_status);
        set_default_nan_mode(false, &lane->fp_status);
    }
}

/* TODO: Implement warp execution (RV32I + RV32F interpreter) */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    (void)s;
    (void)warp;
    (void)max_cycles;
    return 0;
}

int gpgpu_core_exec_kernel(GPGPUState *s)
{
    static const uint32_t max_cycles_per_warp = 100000;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t warp_width;
    uint64_t block_threads;
    uint64_t warps_per_block;
    uint64_t num_blocks;
    uint32_t bz, by, bx;

    if (!s || !s->vram_ptr) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: kernel dispatch without backing VRAM\n");
        return -1;
    }

    if (!(s->global_ctrl & GPGPU_CTRL_ENABLE)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: kernel dispatch while device is disabled\n");
        return -1;
    }

    if (s->kernel.kernel_addr >= s->vram_size ||
        s->kernel.kernel_addr > UINT32_MAX) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: invalid kernel address 0x%" PRIx64 "\n",
                      s->kernel.kernel_addr);
        return -1;
    }

    grid_x = s->kernel.grid_dim[0];
    grid_y = s->kernel.grid_dim[1];
    grid_z = s->kernel.grid_dim[2];
    block_x = s->kernel.block_dim[0];
    block_y = s->kernel.block_dim[1];
    block_z = s->kernel.block_dim[2];

    if (!grid_x || !grid_y || !grid_z || !block_x || !block_y || !block_z) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: invalid grid/block dims grid=(%u,%u,%u) "
                      "block=(%u,%u,%u)\n",
                      grid_x, grid_y, grid_z, block_x, block_y, block_z);
        return -1;
    }

    warp_width = s->warp_size;
    if (!warp_width || warp_width > GPGPU_WARP_SIZE ||
        !is_power_of_2(warp_width)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: invalid warp size %u (max %u, power-of-2 required)\n",
                      warp_width, GPGPU_WARP_SIZE);
        return -1;
    }

    block_threads = (uint64_t)block_x * block_y * block_z;
    warps_per_block = DIV_ROUND_UP(block_threads, warp_width);
    num_blocks = (uint64_t)grid_x * grid_y * grid_z;

    if (block_threads > UINT32_MAX) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: block thread count %" PRIu64 " exceeds 32-bit state\n",
                      block_threads);
        return -1;
    }

    if (warps_per_block > BIT_ULL(MHARTID_WARP_BITS)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: warp count per block %" PRIu64
                      " exceeds mhartid warp field\n",
                      warps_per_block);
        return -1;
    }

    if (num_blocks > BIT_ULL(MHARTID_BLOCK_BITS)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: block count %" PRIu64
                      " exceeds mhartid block field\n",
                      num_blocks);
        return -1;
    }

    if (!warps_per_block) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: block has zero runnable warps\n");
        return -1;
    }

    for (bz = 0; bz < grid_z; bz++) {
        for (by = 0; by < grid_y; by++) {
            for (bx = 0; bx < grid_x; bx++) {
                uint32_t block_id[3] = { bx, by, bz };
                uint32_t block_id_linear = bx + grid_x * (by + grid_y * bz);
                uint64_t warp_idx;

                s->simt.block_id[0] = bx;
                s->simt.block_id[1] = by;
                s->simt.block_id[2] = bz;
                s->simt.barrier_count = 0;
                s->simt.barrier_target = block_threads;
                s->simt.barrier_active = false;

                for (warp_idx = 0; warp_idx < warps_per_block; warp_idx++) {
                    GPGPUWarp warp;
                    uint32_t thread_id_base = warp_idx * warp_width;
                    uint32_t active_threads =
                        MIN((uint64_t)warp_width, block_threads - thread_id_base);

                    gpgpu_core_init_warp(&warp, s->kernel.kernel_addr,
                                         thread_id_base, block_id,
                                         active_threads, warp_idx,
                                         block_id_linear);

                    s->simt.thread_id[0] = thread_id_base;
                    s->simt.thread_id[1] = 0;
                    s->simt.thread_id[2] = 0;
                    s->simt.warp_id = warp_idx;
                    s->simt.lane_id = 0;
                    s->simt.thread_mask = warp.active_mask;

                    if (gpgpu_core_exec_warp(s, &warp, max_cycles_per_warp) < 0) {
                        qemu_log_mask(LOG_GUEST_ERROR,
                                      "gpgpu: warp execution failed for "
                                      "block=(%u,%u,%u) warp=%" PRIu64 "\n",
                                      bx, by, bz, warp_idx);
                        return -1;
                    }
                }
            }
        }
    }

    return 0;
}
