/*
 * QEMU Educational GPGPU Device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"

#include "gpgpu.h"
#include "gpgpu_core.h"

static void gpgpu_reset_state(GPGPUState *s, bool clear_vram)
{
    s->global_ctrl = 0;
    s->global_status = GPGPU_STATUS_READY;
    s->error_status = 0;
    s->irq_enable = 0;
    s->irq_status = 0;
    memset(&s->kernel, 0, sizeof(s->kernel));
    memset(&s->dma, 0, sizeof(s->dma));
    memset(&s->simt, 0, sizeof(s->simt));

    if (s->dma_timer) {
        timer_del(s->dma_timer);
    }

    if (s->kernel_timer) {
        timer_del(s->kernel_timer);
    }
    
    if (clear_vram && s->vram_ptr) {
        memset(s->vram_ptr, 0, s->vram_size);
    }
}

/* BAR0: control register read */
static uint64_t gpgpu_ctrl_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = opaque;
    uint32_t value = 0;

    (void)size;

    switch (addr) {
    case GPGPU_REG_DEV_ID:
        value = GPGPU_DEV_ID_VALUE;
        break;
    case GPGPU_REG_DEV_VERSION:
        value = GPGPU_DEV_VERSION_VALUE;
        break;
    case GPGPU_REG_DEV_CAPS:
        value = (s->num_cus & 0xff) |
                ((s->warps_per_cu & 0xff) << 8) |
                ((s->warp_size & 0xff) << 16);
        break;
    case GPGPU_REG_VRAM_SIZE_LO:
        value = (uint32_t)s->vram_size;
        break;
    case GPGPU_REG_VRAM_SIZE_HI:
        value = (uint32_t)(s->vram_size >> 32);
        break;
    case GPGPU_REG_GLOBAL_CTRL:
        value = s->global_ctrl;
        break;
    case GPGPU_REG_GLOBAL_STATUS:
        value = s->global_status;
        break;
    case GPGPU_REG_ERROR_STATUS:
        value = s->error_status;
        break;
    case GPGPU_REG_IRQ_ENABLE:
        value = s->irq_enable;
        break;
    case GPGPU_REG_IRQ_STATUS:
        value = s->irq_status;
        break;
    case GPGPU_REG_IRQ_ACK:
        value = 0;
        break;
    case GPGPU_REG_KERNEL_ADDR_LO:
        value = (uint32_t)s->kernel.kernel_addr;
        break;
    case GPGPU_REG_KERNEL_ADDR_HI:
        value = (uint32_t)(s->kernel.kernel_addr >> 32);
        break;
    case GPGPU_REG_KERNEL_ARGS_LO:
        value = (uint32_t)s->kernel.kernel_args;
        break;
    case GPGPU_REG_KERNEL_ARGS_HI:
        value = (uint32_t)(s->kernel.kernel_args >> 32);
        break;
    case GPGPU_REG_GRID_DIM_X:
        value = s->kernel.grid_dim[0];
        break;
    case GPGPU_REG_GRID_DIM_Y:
        value = s->kernel.grid_dim[1];
        break;
    case GPGPU_REG_GRID_DIM_Z:
        value = s->kernel.grid_dim[2];
        break;
    case GPGPU_REG_BLOCK_DIM_X:
        value = s->kernel.block_dim[0];
        break;
    case GPGPU_REG_BLOCK_DIM_Y:
        value = s->kernel.block_dim[1];
        break;
    case GPGPU_REG_BLOCK_DIM_Z:
        value = s->kernel.block_dim[2];
        break;
    case GPGPU_REG_SHARED_MEM_SIZE:
        value = s->kernel.shared_mem_size;
        break;
    case GPGPU_REG_DMA_SRC_LO:
        value = (uint32_t)s->dma.src_addr;
        break;
    case GPGPU_REG_DMA_SRC_HI:
        value = (uint32_t)(s->dma.src_addr >> 32);
        break;
    case GPGPU_REG_DMA_DST_LO:
        value = (uint32_t)s->dma.dst_addr;
        break;
    case GPGPU_REG_DMA_DST_HI:
        value = (uint32_t)(s->dma.dst_addr >> 32);
        break;
    case GPGPU_REG_DMA_SIZE:
        value = s->dma.size;
        break;
    case GPGPU_REG_DMA_CTRL:
        value = s->dma.ctrl;
        break;
    case GPGPU_REG_DMA_STATUS:
        value = s->dma.status;
        break;
    case GPGPU_REG_THREAD_ID_X:
        value = s->simt.thread_id[0];
        break;
    case GPGPU_REG_THREAD_ID_Y:
        value = s->simt.thread_id[1];
        break;
    case GPGPU_REG_THREAD_ID_Z:
        value = s->simt.thread_id[2];
        break;
    case GPGPU_REG_BLOCK_ID_X:
        value = s->simt.block_id[0];
        break;
    case GPGPU_REG_BLOCK_ID_Y:
        value = s->simt.block_id[1];
        break;
    case GPGPU_REG_BLOCK_ID_Z:
        value = s->simt.block_id[2];
        break;
    case GPGPU_REG_WARP_ID:
        value = s->simt.warp_id;
        break;
    case GPGPU_REG_LANE_ID:
        value = s->simt.lane_id;
        break;
    case GPGPU_REG_BARRIER:
        value = s->simt.barrier_count;
        break;
    case GPGPU_REG_THREAD_MASK:
        value = s->simt.thread_mask;
        break;
    default:
        break;
    }

    return value;
}

/* BAR0: control register write */
static void gpgpu_ctrl_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = opaque;
    uint32_t value = val;
    
    (void)size;

    switch (addr) {
    case GPGPU_REG_DEV_ID:
    case GPGPU_REG_DEV_VERSION:
    case GPGPU_REG_DEV_CAPS:
    case GPGPU_REG_VRAM_SIZE_LO:
    case GPGPU_REG_VRAM_SIZE_HI:
        break;
    case GPGPU_REG_GLOBAL_CTRL:
        if (value & GPGPU_CTRL_RESET) {
            gpgpu_reset_state(s, false);
        }
        s->global_ctrl = value & GPGPU_CTRL_ENABLE;
       
        break;
    case GPGPU_REG_ERROR_STATUS:
        s->error_status &= ~value;
        break;
    case GPGPU_REG_IRQ_ENABLE:
        s->irq_enable = value;
        break;
    case GPGPU_REG_IRQ_ACK:
        s->irq_status &= ~value;
        break;
    case GPGPU_REG_KERNEL_ADDR_LO:
        s->kernel.kernel_addr =
            (s->kernel.kernel_addr & 0xffffffff00000000ULL) | value;
        break;
    case GPGPU_REG_KERNEL_ADDR_HI:
        s->kernel.kernel_addr =
            (s->kernel.kernel_addr & 0xffffffffULL) | ((uint64_t)value << 32);
        break;
    case GPGPU_REG_KERNEL_ARGS_LO:
        s->kernel.kernel_args =
            (s->kernel.kernel_args & 0xffffffff00000000ULL) | value;
        break;
    case GPGPU_REG_KERNEL_ARGS_HI:
        s->kernel.kernel_args =
            (s->kernel.kernel_args & 0xffffffffULL) | ((uint64_t)value << 32);
        break;
    case GPGPU_REG_GRID_DIM_X:
        s->kernel.grid_dim[0] = value;
        break;
    case GPGPU_REG_GRID_DIM_Y:
        s->kernel.grid_dim[1] = value;
        break;
    case GPGPU_REG_GRID_DIM_Z:
        s->kernel.grid_dim[2] = value;
        break;
    case GPGPU_REG_BLOCK_DIM_X:
        s->kernel.block_dim[0] = value;
        break;
    case GPGPU_REG_BLOCK_DIM_Y:
        s->kernel.block_dim[1] = value;
        break;
    case GPGPU_REG_BLOCK_DIM_Z:
        s->kernel.block_dim[2] = value;
        break;
    case GPGPU_REG_SHARED_MEM_SIZE:
        s->kernel.shared_mem_size = value;
        break;
    case GPGPU_REG_DISPATCH:
        if (s->global_status & GPGPU_STATUS_BUSY) {
            s->error_status |= GPGPU_ERR_INVALID_CMD;
            return;
        }
        
        s->global_status = GPGPU_STATUS_BUSY;
        if (gpgpu_core_exec_kernel(s) == 0) {
            s->global_status = GPGPU_STATUS_READY;
            s->irq_status |= GPGPU_IRQ_KERNEL_DONE;
        } else {
            s->global_status = GPGPU_STATUS_READY | GPGPU_STATUS_ERROR;
            s->error_status |= GPGPU_ERR_KERNEL_FAULT;
            s->irq_status |= GPGPU_IRQ_ERROR;
        }
        break;
    case GPGPU_REG_DMA_SRC_LO:
        s->dma.src_addr = (s->dma.src_addr & 0xffffffff00000000ULL) | value;
        break;
    case GPGPU_REG_DMA_SRC_HI:
        s->dma.src_addr =
            (s->dma.src_addr & 0xffffffffULL) | ((uint64_t)value << 32);
        break;
    case GPGPU_REG_DMA_DST_LO:
        s->dma.dst_addr = (s->dma.dst_addr & 0xffffffff00000000ULL) | value;
        break;
    case GPGPU_REG_DMA_DST_HI:
        s->dma.dst_addr =
            (s->dma.dst_addr & 0xffffffffULL) | ((uint64_t)value << 32);
        break;
    case GPGPU_REG_DMA_SIZE:
        s->dma.size = value;
        break;
    case GPGPU_REG_DMA_CTRL:
        s->dma.ctrl = value;
        break;
    case GPGPU_REG_THREAD_ID_X:
        s->simt.thread_id[0] = value;
        break;
    case GPGPU_REG_THREAD_ID_Y:
        s->simt.thread_id[1] = value;
        break;
    case GPGPU_REG_THREAD_ID_Z:
        s->simt.thread_id[2] = value;
        break;
    case GPGPU_REG_BLOCK_ID_X:
        s->simt.block_id[0] = value;
        break;
    case GPGPU_REG_BLOCK_ID_Y:
        s->simt.block_id[1] = value;
        break;
    case GPGPU_REG_BLOCK_ID_Z:
        s->simt.block_id[2] = value;
        break;
    case GPGPU_REG_WARP_ID:
        s->simt.warp_id = value;
        break;
    case GPGPU_REG_LANE_ID:
        s->simt.lane_id = value;
        break;
    case GPGPU_REG_BARRIER:
        s->simt.barrier_count++;
        break;
    case GPGPU_REG_THREAD_MASK:
        s->simt.thread_mask = value;
        break;
    default:
        break;
    }
    return;
}

static const MemoryRegionOps gpgpu_ctrl_ops = {
    .read = gpgpu_ctrl_read,
    .write = gpgpu_ctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

static uint64_t gpgpu_vram_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = opaque;
    uint64_t value = 0;

    if (!s->vram_ptr || size == 0 || addr + size > s->vram_size) {
        s->global_status |= GPGPU_STATUS_ERROR;
        s->error_status |= GPGPU_ERR_VRAM_FAULT;
        s->irq_status |= GPGPU_IRQ_ERROR;
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: vram read out of range addr=0x%" HWADDR_PRIx
                      " size=%u vram_size=0x%" PRIx64 "\n",
                      addr, size, s->vram_size);
        return 0;
    }

    for (unsigned i = 0; i < size; i++) {
        value |= ((uint64_t)s->vram_ptr[addr + i]) << (i * 8);
    }

    return value;
}

static void gpgpu_vram_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = opaque;

    if (!s->vram_ptr || size == 0 || addr + size > s->vram_size) {
        s->global_status |= GPGPU_STATUS_ERROR;
        s->error_status |= GPGPU_ERR_VRAM_FAULT;
        s->irq_status |= GPGPU_IRQ_ERROR;
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: vram write out of range addr=0x%" HWADDR_PRIx
                      " size=%u vram_size=0x%" PRIx64 "\n",
                      addr, size, s->vram_size);
        return;
    }

    for (unsigned i = 0; i < size; i++) {
        s->vram_ptr[addr + i] = (uint8_t)(val >> (i * 8));
    }
}

static const MemoryRegionOps gpgpu_vram_ops = {
    .read = gpgpu_vram_read,
    .write = gpgpu_vram_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static uint64_t gpgpu_doorbell_read(void *opaque, hwaddr addr, unsigned size)
{
    return 0;
}

static void gpgpu_doorbell_write(void *opaque, hwaddr addr, uint64_t val,
                                 unsigned size)
{
    
    // #define GPGPU_CTRL_BAR_SIZE     (1 * 1024 * 1024)   /* BAR0: 控制寄存器 1MB */
    // #define GPGPU_VRAM_BAR_SIZE     (64 * 1024 * 1024)  /* BAR2: 显存 64MB (默认) */
    // #define GPGPU_DOORBELL_BAR_SIZE (64 * 1024)         /* BAR4: 门铃寄存器 64KB */

   
    return;

    
}

static const MemoryRegionOps gpgpu_doorbell_ops = {
    .read = gpgpu_doorbell_read,
    .write = gpgpu_doorbell_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* TODO: Implement DMA completion handler */
static void gpgpu_dma_complete(void *opaque)
{
    (void)opaque;
}

/* TODO: Implement kernel completion handler */
static void gpgpu_kernel_complete(void *opaque)
{
    (void)opaque;
}

static void gpgpu_realize(PCIDevice *pdev, Error **errp)
{
    GPGPUState *s = GPGPU(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    s->vram_ptr = g_malloc0(s->vram_size);
    if (!s->vram_ptr) {
        error_setg(errp, "GPGPU: failed to allocate VRAM");
        return;
    }

    /* BAR 0: control registers */
    memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ctrl_ops, s,
                          "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);
    pci_register_bar(pdev, 0,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64,
                     &s->ctrl_mmio);

    /* BAR 2: VRAM */
    memory_region_init_io(&s->vram, OBJECT(s), &gpgpu_vram_ops, s,
                          "gpgpu-vram", s->vram_size);
    pci_register_bar(pdev, 2,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64 |
                     PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vram);

    /* BAR 4: doorbell registers */
    memory_region_init_io(&s->doorbell_mmio, OBJECT(s), &gpgpu_doorbell_ops, s,
                          "gpgpu-doorbell", GPGPU_DOORBELL_BAR_SIZE);
    pci_register_bar(pdev, 4,
                     PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->doorbell_mmio);

    if (msix_init(pdev, GPGPU_MSIX_VECTORS,
                  &s->ctrl_mmio, 0, 0xFE000,
                  &s->ctrl_mmio, 0, 0xFF000,
                  0, errp)) {
        g_free(s->vram_ptr);
        return;
    }

    msi_init(pdev, 0, 1, true, false, errp);

    s->dma_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, gpgpu_dma_complete, s);
    s->kernel_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,
                                   gpgpu_kernel_complete, s);

    gpgpu_reset_state(s, true);
}

static void gpgpu_exit(PCIDevice *pdev)
{
    GPGPUState *s = GPGPU(pdev);

    timer_free(s->dma_timer);
    timer_free(s->kernel_timer);
    g_free(s->vram_ptr);
    msix_uninit(pdev, &s->ctrl_mmio, &s->ctrl_mmio);
    msi_uninit(pdev);
}

static void gpgpu_reset(DeviceState *dev)
{
    GPGPUState *s = GPGPU(dev);

    gpgpu_reset_state(s, true);
}

static const Property gpgpu_properties[] = {
    DEFINE_PROP_UINT32("num_cus", GPGPUState, num_cus,
                       GPGPU_DEFAULT_NUM_CUS),
    DEFINE_PROP_UINT32("warps_per_cu", GPGPUState, warps_per_cu,
                       GPGPU_DEFAULT_WARPS_PER_CU),
    DEFINE_PROP_UINT32("warp_size", GPGPUState, warp_size,
                       GPGPU_DEFAULT_WARP_SIZE),
    DEFINE_PROP_UINT64("vram_size", GPGPUState, vram_size,
                       GPGPU_DEFAULT_VRAM_SIZE),
};

static const VMStateDescription vmstate_gpgpu = {
    .name = "gpgpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, GPGPUState),
        VMSTATE_UINT32(global_ctrl, GPGPUState),
        VMSTATE_UINT32(global_status, GPGPUState),
        VMSTATE_UINT32(error_status, GPGPUState),
        VMSTATE_UINT32(irq_enable, GPGPUState),
        VMSTATE_UINT32(irq_status, GPGPUState),
        VMSTATE_END_OF_LIST()
    }
};

static void gpgpu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->realize = gpgpu_realize;
    pc->exit = gpgpu_exit;
    pc->vendor_id = GPGPU_VENDOR_ID;
    pc->device_id = GPGPU_DEVICE_ID;
    pc->revision = GPGPU_REVISION;
    pc->class_id = GPGPU_CLASS_CODE;

    device_class_set_legacy_reset(dc, gpgpu_reset);
    dc->desc = "Educational GPGPU Device";
    dc->vmsd = &vmstate_gpgpu;
    device_class_set_props(dc, gpgpu_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo gpgpu_type_info = {
    .name          = TYPE_GPGPU,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init    = gpgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { }
    },
};

static void gpgpu_register_types(void)
{
    type_register_static(&gpgpu_type_info);
}

type_init(gpgpu_register_types)
