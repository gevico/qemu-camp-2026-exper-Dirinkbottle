#ifndef HW_WDT_G233_WDT_H
#define HW_WDT_G233_WDT_H

/*
 * Educational watchdog timer skeleton for the G233 board.
 *
 * Scope of this first step:
 * - expose one MMIO page on the system bus
 * - expose one aggregated interrupt output towards the board IRQ chip/PLIC
 * - keep the guest-visible register contract stable
 *
 * Intentionally left for later:
 * - virtual-time countdown modelling
 * - timeout scheduling
 * - reset output behaviour when RSTEN is set
 *
 * The board model is responsible for:
 * 1. mapping region 0 into the machine memmap
 * 2. connecting irq 0 to the chosen PLIC source
 * 3. describing the block in the generated DT
 */

#include "hw/core/irq.h"
#include "hw/core/sysbus.h"
#include "qom/object.h"
#include "hw/core/ptimer.h"

#define TYPE_G233_WDT "g233-wdt"
OBJECT_DECLARE_SIMPLE_TYPE(G233WDTState, G233_WDT)

#define G233_WDT_MMIO_SIZE        0x1000

/* Guest-visible register offsets */
#define G233_WDT_REG_CTRL         0x00
#define G233_WDT_REG_LOAD         0x04
#define G233_WDT_REG_VAL          0x08
#define G233_WDT_REG_SR           0x10
#define G233_WDT_REG_KEY          0x0c

/* WDT_CTRL bits */
#define G233_WDT_CTRL_EN          (1u << 0)
#define G233_WDT_CTRL_INTEN       (1u << 1)
#define G233_WDT_CTRL_RSTEN       (1u << 2)
#define G233_WDT_CTRL_LOCK        (1u << 3)
#define G233_WDT_CTRL_WRMASK      (G233_WDT_CTRL_EN | \
                                   G233_WDT_CTRL_INTEN | \
                                   G233_WDT_CTRL_RSTEN)

/* WDT_SR bits */
#define G233_WDT_SR_TIMEOUT       (1u << 0)

/* WDT_KEY values */
#define G233_WDT_KEY_FEED         0x5A5A5A5Au
#define G233_WDT_KEY_LOCK         0x1ACCE551u


/* WDT timer default hz */
#define G233_WDT_TIMER_HZ 10000

typedef struct G233WDTState {
    SysBusDevice parent_obj;

    MemoryRegion mmio;
    qemu_irq irq;


    /* self timer ptimer */
    struct ptimer_state *timer;

    /*
     * Guest-visible register storage.
     *
     * ctrl:
     *   stores writable control bits EN/INTEN/RSTEN
     *
     * load:
     *   reload value programmed by software
     *
     * value:
     *   current counter snapshot used by this scaffold; later you can replace
     *   it with a virtual-time derived helper without changing the MMIO API
     *
     * status:
     *   sticky status bits such as TIMEOUT
     *
     * locked:
     *   board-visible LOCK state, reflected as CTRL[3] on reads
     */
    uint32_t ctrl;
    uint32_t load;
    uint32_t value;
    uint32_t status;
    bool locked;
} G233WDTState;

#endif /* HW_WDT_G233_WDT_H */
