#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "migration/vmstate.h"
#include "hw/core/qdev-properties.h"
#include "hw/pwmctrl/g233_pwmctrl.h"

/*
 * Learning roadmap for this device:
 *
 * 1. MMIO contract
 *    Read test-pwm-basic.c first. That file defines what the guest expects.
 *
 * 2. Time modelling
 *    QEMU does not need a host loop that increments CNT every tick.
 *    A better model stores a start timestamp and derives CNT from virtual time.
 *
 * 3. Period boundary event
 *    The host-side QEMUTimer is used to wake the device at the next channel
 *    wrap point so DONE/IRQ/output state can be refreshed.
 *
 * 4. Waveform semantics
 *    The scaffold already computes a simple logical output level from CNT,
 *    DUTY and POL. If your hardware differs, adjust the helper in one place.
 *
 * 5. Board integration
 *    This file models the controller. The board file decides where the MMIO
 *    window lives and which PLIC source receives the aggregated IRQ.
 */

static inline uint64_t g233_pwmctrl_ns_to_ticks(G233PWMCtrlState *s,
                                                uint64_t ns)
{
    if (!s->clock_frequency) {
        return 0;
    }

    return muldiv64(ns, s->clock_frequency, NANOSECONDS_PER_SECOND);
}

static inline uint64_t g233_pwmctrl_ticks_to_ns(G233PWMCtrlState *s,
                                                uint64_t ticks)
{
    if (!s->clock_frequency) {
        return 0;
    }

    return muldiv64(ticks, NANOSECONDS_PER_SECOND, s->clock_frequency);
}

static bool g233_pwmctrl_decode_channel_reg(hwaddr offset,
                                            unsigned int *channel,
                                            hwaddr *channel_reg)
{
    hwaddr local = offset - G233_PWM_CH_BASE(0);

    if (offset < G233_PWM_CH_BASE(0)) {
        return false;
    }

    *channel = local / G233_PWM_CH_STRIDE;
    *channel_reg = local % G233_PWM_CH_STRIDE;

    return *channel < G233_PWMCTRL_NUM_CHANNELS;
}

/*
 * Return the software-visible CNT value for one channel.
 *
 * Current scaffold policy:
 * - if the channel is stopped, return the frozen counter_latch
 * - if the channel is running and PERIOD != 0, derive CNT from virtual time
 *   and wrap inside the period
 * - if PERIOD == 0, treat the channel as non-advancing
 *
 * TODO(user):
 * Decide whether your hardware treats PERIOD=0 as:
 * - fully disabled
 * - a 1-tick period
 * - undefined / no waveform
 */
static uint32_t g233_pwmctrl_get_counter(G233PWMCtrlState *s,
                                         unsigned int channel)
{
    uint64_t now_ns;
    uint64_t elapsed_ticks;
    uint64_t period;

    if (!s->running[channel]) {
        return s->counter_latch[channel];
    }

    period = s->period[channel];
    if (period == 0) {
        return s->counter_latch[channel];
    }

    now_ns = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    elapsed_ticks = g233_pwmctrl_ns_to_ticks(s, now_ns -
                                             s->counter_origin_ns[channel]);
    /*
     * TODO(user):
     * Replace this placeholder with the real counter evolution rule.
     *
     * Typical periodic PWM choice:
     *   return (counter_latch + elapsed_ticks) % period;
     *
     * Alternative one-shot choice:
     *   return MIN(counter_latch + elapsed_ticks, period - 1);
     *
     * The skeleton intentionally returns the frozen latch so register/MMIO/board
     * integration can compile before you decide the hardware semantics.
     */
    return ((s->counter_latch[channel] + elapsed_ticks ) % s->period[channel]);
        


}

/*
 * Freeze the current time-derived counter into counter_latch[].
 *
 * This helper is called before register writes that change the channel state.
 * It lets the model transition between "running in virtual time" and "stopped
 * with a stable count" without losing the guest-visible CNT value.
 */
static void g233_pwmctrl_sync_counter(G233PWMCtrlState *s,
                                      unsigned int channel)
{
    s->counter_latch[channel] = g233_pwmctrl_get_counter(s, channel);
    s->counter_origin_ns[channel] = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
}

static uint32_t g233_pwmctrl_glb_en_bits(G233PWMCtrlState *s)
{
    uint32_t bits = 0;
    unsigned int channel;

    for (channel = 0; channel < G233_PWMCTRL_NUM_CHANNELS; channel++) {
        if (s->ctrl[channel] & G233_PWM_CTRL_EN) {
            bits |= G233_PWM_GLB_CH_EN(channel);
        }
    }

    return bits;
}

/*
 * Compute the current logical waveform level of one channel.
 *
 * Current scaffold policy:
 * - EN=0      -> output low
 * - PERIOD=0  -> output low
 * - active window is CNT < min(DUTY, PERIOD)
 * - POL=0 means non-inverted, POL=1 means inverted
 *
 * TODO(user):
 * If your real PWM block has different semantics, this is the function to
 * replace. Common variations:
 * - idle level follows POL when EN=0
 * - DUTY == PERIOD means constant-high instead of "always active window"
 * - center-aligned or one-shot modes need more state than CNT<duty
 */
static bool g233_pwmctrl_get_output_level(G233PWMCtrlState *s,
                                          unsigned int channel)
{
    uint32_t cnt;
    uint32_t period;
    uint32_t duty;
    bool active;
    bool invert;

    if (!(s->ctrl[channel] & G233_PWM_CTRL_EN)) {
        return false;
    }

    period = s->period[channel];
    if (period == 0) {
        return false;
    }

    cnt = g233_pwmctrl_get_counter(s, channel);
    duty = MIN(s->duty[channel], period);
    active = cnt < duty;
    invert = (s->ctrl[channel] & G233_PWM_CTRL_POL) != 0;
    /*
     * TODO(user):
     * Replace this placeholder with the actual waveform rule.
     *
     * Common edge-aligned rule:
     *   return invert ? !active : active;
     *
     * Some hardware instead defines POL as idle-level select, or forces a
     * constant level when DUTY == 0 / DUTY >= PERIOD.
     */
    bool level = active ? !invert : invert;
    return level;
}

static void g233_pwmctrl_update_outputs(G233PWMCtrlState *s)
{
    unsigned int channel;

    for (channel = 0; channel < G233_PWMCTRL_NUM_CHANNELS; channel++) {
        bool level = g233_pwmctrl_get_output_level(s, channel);

        /* update when change */
        if (level != s->last_output_level[channel]) {
            s->last_output_level[channel] = level;
            qemu_set_irq(s->pwm_out[channel], level);
        }
    }
}

/*
 * Aggregate device interrupt state.
 *
 * Current scaffold policy:
 * - any sticky DONE bit raises the single exported IRQ line
 *
 * TODO(user):
 * If your manual later introduces a separate interrupt enable/mask register,
 * implement the filtering here instead of changing the board wiring.
 */
static void g233_pwmctrl_update_irq(G233PWMCtrlState *s)
{
    qemu_set_irq(s->irq, (s->done_bits & G233_PWM_GLB_DONE_MASK) != 0);
}

/*
 * Schedule the next "period complete" wakeup for one channel.
 *
 * The timer is host-side infrastructure only. It lets the model wake up at the
 * next wrap boundary without polling.
 */
static void g233_pwmctrl_schedule_channel(G233PWMCtrlState *s,
                                          unsigned int channel)
{
    uint64_t now_ns;
    uint64_t period;
    uint64_t cnt;
    uint64_t ticks_until_wrap;
    uint64_t delta_ns;

    if (!s->running[channel] || !(s->ctrl[channel] & G233_PWM_CTRL_EN)) {
        timer_del(&s->done_timer[channel]);
        return;
    }

    period = s->period[channel];
    if (period == 0 || !s->clock_frequency) {
        timer_del(&s->done_timer[channel]);
        return;
    }

    cnt = g233_pwmctrl_get_counter(s, channel);
    ticks_until_wrap = period - cnt;
    if (ticks_until_wrap == 0) {
        ticks_until_wrap = period;
    }

    delta_ns = g233_pwmctrl_ticks_to_ns(s, ticks_until_wrap);
    if (delta_ns == 0) {
        delta_ns = 1;
    }

    now_ns = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    /*
     * TODO(user):
     * Decide what event should wake the device next.
     *
     * For a basic periodic PWM, the next event is usually the wrap boundary:
     *   timer_mod(&done_timer[channel], now_ns + delta_ns);
     *
     * If you later model compare edges too, you may need a second timer or a
     * choice between "next duty edge" and "next period edge".
     */

     timer_mod(&s->done_timer[channel], now_ns + delta_ns);

     return;
}

/*
 * Handle one channel reaching a period boundary.
 *
 * Current scaffold policy:
 * - set sticky DONE bit
 * - restart timing from CNT=0 if the channel remains enabled
 * - keep running periodically until the guest disables the channel
 *
 * TODO(user):
 * Decide whether your hardware is:
 * - periodic level generator
 * - one-shot pulse generator
 * - periodic generator with auto-clear/auto-reload side effects
 */
static void g233_pwmctrl_channel_done(G233PWMCtrlState *s,
                                      unsigned int channel)
{
    if (!(s->ctrl[channel] & G233_PWM_CTRL_EN) || !s->running[channel]) {
        return;
    }

    /*
     * TODO(user):
     * This is the host callback for a scheduled channel event.
     *
     * Typical things to decide here:
     * - Should CH_DONE become sticky?
     * - Should CNT wrap to 0 or stop at PERIOD-1?
     * - Should EN auto-clear in one-shot mode?
     * - Should the next event be scheduled immediately for periodic mode?
     *
     * Minimal periodic example:
     *   done_bits |= CH_DONE(channel);
     *   counter_latch[channel] = 0;
     *   counter_origin_ns[channel] = now;
     *   update_outputs();
     *   update_irq();
     *   schedule_channel();
     */
    s->done_bits |= G233_PWM_GLB_CH_DONE(channel);
    s->counter_latch[channel] = 0 ;
    s->counter_origin_ns[channel] = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

    g233_pwmctrl_update_outputs(s);
    g233_pwmctrl_update_irq(s);
    g233_pwmctrl_schedule_channel(s, channel);

    return;
}

static void g233_pwmctrl_done_timer_cb(void *opaque)
{
    G233PWMCtrlTimerCtx *ctx = opaque;

    g233_pwmctrl_channel_done(ctx->s, ctx->channel);
}

static void g233_pwmctrl_channel_set_enable(G233PWMCtrlState *s,
                                            unsigned int channel,
                                            bool enable)
{
    if (enable == s->running[channel]) {
        g233_pwmctrl_schedule_channel(s, channel);
        return;
    }

    g233_pwmctrl_sync_counter(s, channel);
    s->running[channel] = enable;


    
    if (enable) {
        s->counter_latch[channel] = 0;
        uint64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
        s->counter_origin_ns[channel] = now;
        g233_pwmctrl_schedule_channel(s, channel);
    } else {
        timer_del(&s->done_timer[channel]);
    }

    /*
     * Current scaffold resumes from the frozen counter when re-enabled.
     * TODO(user): if the real hardware restarts from zero on EN rising edge,
     * clear counter_latch[channel] here before setting counter_origin_ns.
     */
    s->counter_origin_ns[channel] = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    /*
     * TODO(user):
     * Decide what EN transitions mean.
     *
     * Common choices:
     * - pause/resume: keep counter_latch, keep progress point
     * - restart: set counter_latch = 0 when EN rises
     * - one-shot arm: schedule exactly one period-complete event
     */
}

static uint64_t g233_pwmctrl_read(void *opaque, hwaddr offset, unsigned size)
{
    G233PWMCtrlState *s = opaque;
    unsigned int channel;
    hwaddr channel_reg;

    if (size != 4) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: unsupported read size %u at 0x%" HWADDR_PRIx "\n",
                      TYPE_G233_PWMCTRL, size, offset);
        return 0;
    }

    switch (offset) {
    case G233_PWM_REG_GLB:
        return g233_pwmctrl_glb_en_bits(s) |
               (s->done_bits & G233_PWM_GLB_DONE_MASK);
    default:
        break;
    }

    if (!g233_pwmctrl_decode_channel_reg(offset, &channel, &channel_reg)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: bad read offset 0x%" HWADDR_PRIx "\n",
                      TYPE_G233_PWMCTRL, offset);
        return 0;
    }

    switch (channel_reg) {
    case G233_PWM_REG_CH_CTRL:
        return s->ctrl[channel];
    case G233_PWM_REG_CH_PERIOD:
        return s->period[channel];
    case G233_PWM_REG_CH_DUTY:
        return s->duty[channel];
    case G233_PWM_REG_CH_CNT:
        return g233_pwmctrl_get_counter(s, channel);
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: bad channel read offset 0x%" HWADDR_PRIx "\n",
                      TYPE_G233_PWMCTRL, offset);
        return 0;
    }
}

static void g233_pwmctrl_write(void *opaque, hwaddr offset,
                               uint64_t value, unsigned size)
{
    G233PWMCtrlState *s = opaque;
    uint32_t data = value;
    unsigned int channel;
    hwaddr channel_reg;
    bool old_enable;
    bool new_enable;

    if (size != 4) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: unsupported write size %u at 0x%" HWADDR_PRIx "\n",
                      TYPE_G233_PWMCTRL, size, offset);
        return;
    }

    if (offset == G233_PWM_REG_GLB) {
        /*
         * Lower EN bits are mirrors, not writable configuration in this model.
         * Upper DONE bits are sticky W1C flags.
         */
        s->done_bits &= ~(data & G233_PWM_GLB_DONE_MASK);
        g233_pwmctrl_update_irq(s);
        return;
    }

    if (!g233_pwmctrl_decode_channel_reg(offset, &channel, &channel_reg)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: bad write offset 0x%" HWADDR_PRIx " = 0x%" PRIx64 "\n",
                      TYPE_G233_PWMCTRL, offset, value);
        return;
    }

    switch (channel_reg) {
    case G233_PWM_REG_CH_CTRL:
        old_enable = (s->ctrl[channel] & G233_PWM_CTRL_EN) != 0;
        s->ctrl[channel] = data & G233_PWM_CTRL_VALID_MASK;
        new_enable = (s->ctrl[channel] & G233_PWM_CTRL_EN) != 0;
        if (old_enable != new_enable) {
            g233_pwmctrl_channel_set_enable(s, channel, new_enable);
        } else if (s->running[channel]) {
            g233_pwmctrl_sync_counter(s, channel);
            g233_pwmctrl_schedule_channel(s, channel);
        }
        break;
    case G233_PWM_REG_CH_PERIOD:
        g233_pwmctrl_sync_counter(s, channel);
        s->period[channel] = data;
        if (s->period[channel] != 0) {
            s->counter_latch[channel] %= s->period[channel];
        } else {
            s->counter_latch[channel] = 0;
        }
        g233_pwmctrl_schedule_channel(s, channel);
        break;
    case G233_PWM_REG_CH_DUTY:
        /*
         * Keep the raw guest value. Output helpers clamp against PERIOD when
         * deriving the waveform. This is a common modelling trick that keeps
         * register readback faithful while still avoiding invalid array maths.
         *
         * TODO(user): if the real hardware saturates DUTY in the register
         * itself, clamp s->duty[channel] at write time instead.
         */
        s->duty[channel] = data;
        break;
    case G233_PWM_REG_CH_CNT:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: write to read-only CNT register ignored (ch%u)\n",
                      TYPE_G233_PWMCTRL, channel);
        return;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: bad channel write offset 0x%" HWADDR_PRIx
                      " = 0x%" PRIx64 "\n",
                      TYPE_G233_PWMCTRL, offset, value);
        return;
    }

    g233_pwmctrl_update_outputs(s);
    g233_pwmctrl_update_irq(s);
}

static const MemoryRegionOps g233_pwmctrl_ops = {
    .read = g233_pwmctrl_read,
    .write = g233_pwmctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

static void g233_pwmctrl_reset(DeviceState *dev)
{
    G233PWMCtrlState *s = G233_PWMCTRL(dev);
    unsigned int channel;

    s->done_bits = 0;

    for (channel = 0; channel < G233_PWMCTRL_NUM_CHANNELS; channel++) {
        s->ctrl[channel] = 0;
        s->period[channel] = 0;
        s->duty[channel] = 0;
        s->counter_latch[channel] = 0;
        s->counter_origin_ns[channel] = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
        s->running[channel] = false;
        s->last_output_level[channel] = false;
        timer_del(&s->done_timer[channel]);
    }

    g233_pwmctrl_update_outputs(s);
    g233_pwmctrl_update_irq(s);
}

static void g233_pwmctrl_init(Object *obj)
{
    G233PWMCtrlState *s = G233_PWMCTRL(obj);
    DeviceState *dev = DEVICE(obj);

    memory_region_init_io(&s->mmio, obj, &g233_pwmctrl_ops, s,
                          TYPE_G233_PWMCTRL, G233_PWMCTRL_MMIO_SIZE);
    sysbus_init_mmio(SYS_BUS_DEVICE(obj), &s->mmio);
    sysbus_init_irq(SYS_BUS_DEVICE(obj), &s->irq);

    qdev_init_gpio_out(dev, s->pwm_out, G233_PWMCTRL_NUM_CHANNELS);
}

static void g233_pwmctrl_realize(DeviceState *dev, Error **errp)
{
    G233PWMCtrlState *s = G233_PWMCTRL(dev);
    unsigned int channel;

    for (channel = 0; channel < G233_PWMCTRL_NUM_CHANNELS; channel++) {
        s->timer_ctx[channel].s = s;
        s->timer_ctx[channel].channel = channel;
        timer_init_ns(&s->done_timer[channel], QEMU_CLOCK_VIRTUAL,
                      g233_pwmctrl_done_timer_cb, &s->timer_ctx[channel]);
    }
}

static const VMStateDescription vmstate_g233_pwmctrl = {
    .name = TYPE_G233_PWMCTRL,
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_TIMER_ARRAY(done_timer, G233PWMCtrlState,
                            G233_PWMCTRL_NUM_CHANNELS),
        VMSTATE_UINT32_ARRAY(ctrl, G233PWMCtrlState, G233_PWMCTRL_NUM_CHANNELS),
        VMSTATE_UINT32_ARRAY(period, G233PWMCtrlState, G233_PWMCTRL_NUM_CHANNELS),
        VMSTATE_UINT32_ARRAY(duty, G233PWMCtrlState, G233_PWMCTRL_NUM_CHANNELS),
        VMSTATE_UINT32(done_bits, G233PWMCtrlState),
        VMSTATE_UINT32_ARRAY(counter_latch, G233PWMCtrlState,
                             G233_PWMCTRL_NUM_CHANNELS),
        VMSTATE_UINT64_ARRAY(counter_origin_ns, G233PWMCtrlState,
                             G233_PWMCTRL_NUM_CHANNELS),
        VMSTATE_BOOL_ARRAY(running, G233PWMCtrlState, G233_PWMCTRL_NUM_CHANNELS),
        VMSTATE_BOOL_ARRAY(last_output_level, G233PWMCtrlState,
                           G233_PWMCTRL_NUM_CHANNELS),
        VMSTATE_UINT64(clock_frequency, G233PWMCtrlState),
        VMSTATE_END_OF_LIST()
    },
};

static const Property g233_pwmctrl_properties[] = {
    DEFINE_PROP_UINT64("clock-frequency", G233PWMCtrlState,
                       clock_frequency, 1000000ULL),
};

static void g233_pwmctrl_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = g233_pwmctrl_realize;
    dc->vmsd = &vmstate_g233_pwmctrl;
    device_class_set_legacy_reset(dc, g233_pwmctrl_reset);
    device_class_set_props(dc, g233_pwmctrl_properties);
}

static const TypeInfo g233_pwmctrl_info = {
    .name = TYPE_G233_PWMCTRL,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(G233PWMCtrlState),
    .instance_init = g233_pwmctrl_init,
    .class_init = g233_pwmctrl_class_init,
};

static void g233_pwmctrl_register_types(void)
{
    type_register_static(&g233_pwmctrl_info);
}

type_init(g233_pwmctrl_register_types)
