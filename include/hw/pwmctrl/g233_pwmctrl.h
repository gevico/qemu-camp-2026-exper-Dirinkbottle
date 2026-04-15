#ifndef HW_PWMCTRL_G233_PWMCTRL_H
#define HW_PWMCTRL_G233_PWMCTRL_H

/*
 * Educational PWM controller skeleton for the G233 board.
 *
 * This device intentionally models the board-visible structure first:
 * - one MMIO register bank
 * - one aggregated interrupt output towards the PLIC/irqchip
 * - four PWM waveform outputs towards pinctrl or direct board consumers
 * - one virtual-clock based time base for each PWM channel
 *
 * The goal is to let the user fill in the hardware semantics step by step.
 * The framework already gives you:
 * - QOM/SysBus device registration
 * - MMIO plumbing
 * - IRQ plumbing
 * - per-channel state storage
 * - QEMU timers for "period complete" style events
 * - migration boilerplate
 *
 * Typical board responsibilities after creating this device:
 * 1. map region 0 into the board memmap with sysbus_mmio_map()
 * 2. connect irq 0 to a PLIC source with sysbus_connect_irq()
 * 3. optionally route pwm_out[] to a pinctrl block, LEDs, buzzers, fans, etc.
 */

#include "hw/core/irq.h"
#include "hw/core/sysbus.h"
#include "qemu/timer.h"
#include "qom/object.h"

#define TYPE_G233_PWMCTRL "g233-pwmctrl"
OBJECT_DECLARE_SIMPLE_TYPE(G233PWMCtrlState, G233_PWMCTRL)

#define G233_PWMCTRL_NUM_CHANNELS   4
#define G233_PWMCTRL_MMIO_SIZE      0x1000

/* Global register block */
#define G233_PWM_REG_GLB            0x00

/* Channel register layout: 0x10 + ch * 0x10 */
#define G233_PWM_CH_STRIDE          0x10
#define G233_PWM_CH_BASE(ch)        (0x10 + (ch) * G233_PWM_CH_STRIDE)
#define G233_PWM_REG_CH_CTRL        0x00
#define G233_PWM_REG_CH_PERIOD      0x04
#define G233_PWM_REG_CH_DUTY        0x08
#define G233_PWM_REG_CH_CNT         0x0c

/* PWM_GLB bits */
#define G233_PWM_GLB_CH_EN(ch)      (1u << (ch))
#define G233_PWM_GLB_CH_DONE(ch)    (1u << (4 + (ch)))
#define G233_PWM_GLB_DONE_MASK      0x000000f0u

/* PWM_CH_CTRL bits */
#define G233_PWM_CTRL_EN            (1u << 0)
#define G233_PWM_CTRL_POL           (1u << 1)
#define G233_PWM_CTRL_VALID_MASK    (G233_PWM_CTRL_EN | G233_PWM_CTRL_POL)

typedef struct G233PWMCtrlTimerCtx {
    struct G233PWMCtrlState *s;
    unsigned int channel;
} G233PWMCtrlTimerCtx;

typedef struct G233PWMCtrlState {
    SysBusDevice parent_obj;

    MemoryRegion mmio;

    /*
     * Single aggregated interrupt line towards the board interrupt controller.
     * The current scaffold raises it when any channel has a sticky DONE bit.
     * TODO(user): if your real hardware has a separate interrupt mask/status
     * layer, add the corresponding registers and change the aggregation logic.
     */
    qemu_irq irq;

    /*
     * Board-facing waveform outputs.
     * Each line represents the current logical PWM output after CTRL/PERIOD/
     * DUTY/POL are interpreted.
     */
    qemu_irq pwm_out[G233_PWMCTRL_NUM_CHANNELS];

    /*
     * One QEMU timer per channel. These do not represent hardware registers;
     * they are host-side scheduling objects used to wake the device when a
     * channel reaches an interesting time boundary.
     */
    QEMUTimer done_timer[G233_PWMCTRL_NUM_CHANNELS];
    G233PWMCtrlTimerCtx timer_ctx[G233_PWMCTRL_NUM_CHANNELS];

    /*
     * Teaching register bank expected by tests/documentation.
     * ctrl/period/duty are guest-visible storage.
     * done_bits backs the sticky CHn_DONE status in PWM_GLB[7:4].
     */
    uint32_t ctrl[G233_PWMCTRL_NUM_CHANNELS];
    uint32_t period[G233_PWMCTRL_NUM_CHANNELS];
    uint32_t duty[G233_PWMCTRL_NUM_CHANNELS];
    uint32_t done_bits;

    /*
     * Time-base bookkeeping.
     *
     * counter_latch[ch]:
     *   software-visible count captured while the channel is not advancing
     *
     * counter_origin_ns[ch]:
     *   virtual timestamp from which the current running interval is measured
     *
     * running[ch]:
     *   host-side notion of whether the channel is currently advancing
     *
     * last_output_level[ch]:
     *   cached output line level so qemu_set_irq() is only called on changes
     */
    uint32_t counter_latch[G233_PWMCTRL_NUM_CHANNELS];
    uint64_t counter_origin_ns[G233_PWMCTRL_NUM_CHANNELS];
    bool running[G233_PWMCTRL_NUM_CHANNELS];
    bool last_output_level[G233_PWMCTRL_NUM_CHANNELS];

    /*
     * Input clock driving the PWM counters.
     * The default is intentionally modest to make qtest clock stepping easy to
     * reason about. Board code can override it with qdev properties later.
     */
    uint64_t clock_frequency;
} G233PWMCtrlState;

#endif /* HW_PWMCTRL_G233_PWMCTRL_H */
