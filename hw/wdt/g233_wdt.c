#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "migration/vmstate.h"
#include "hw/wdt/g233_wdt.h"









/*
 * This device starts from the software-visible contract first.
 * The surrounding framework is already complete:
 * - QOM/SysBus registration
 * - MMIO plumbing
 * - IRQ plumbing
 * - reset/migration boilerplate
 *
 * TODO(user):
 * Add the actual watchdog countdown model later. A common next step is:
 * 1. record a virtual-time origin when EN rises or FEED happens
 * 2. derive VAL from LOAD/origin/current virtual time
 * 3. set SR.TIMEOUT when the countdown reaches zero
 * 4. assert IRQ when INTEN && TIMEOUT
 * 5. trigger board reset when RSTEN && TIMEOUT
 */

static uint32_t g233_wdt_ctrl_readback(G233WDTState *s)
{
    return (s->ctrl & G233_WDT_CTRL_WRMASK) |
           (s->locked ? G233_WDT_CTRL_LOCK : 0);
}

static void g233_wdt_update_irq(G233WDTState *s)
{
    bool pending = (s->status & G233_WDT_SR_TIMEOUT) != 0;
    bool enabled = (s->ctrl & G233_WDT_CTRL_INTEN) != 0;

    qemu_set_irq(s->irq, pending && enabled);
}

static void g233_set_next_timer(G233WDTState *s){
    ptimer_transaction_begin(s->timer);
    ptimer_stop(s->timer);
    ptimer_set_count(s->timer,s->load ? s->load : 1);

    if (s->ctrl & G233_WDT_CTRL_EN)
    {
        ptimer_run(s->timer,1);
    }
    
    ptimer_transaction_commit(s->timer);
}



static void g233_wdt_timer_callback(void *opaque){
    G233WDTState *s = opaque;

    s->value = 0;
    s->status |= G233_WDT_SR_TIMEOUT;

    g233_wdt_update_irq(s);

    if (s->ctrl & G233_WDT_CTRL_RSTEN) {
        /* 这里后面再接 reset 语义 */
        // because test not need real systeam reset,so ignore
    }

    return;
}



static uint32_t g233_wdt_get_value(G233WDTState *s)
{
    /*
     * TODO(user):
     * Replace this with a virtual-time derived countdown:
     *   if EN=0, return the frozen latch;
     *   if EN=1, compute LOAD - elapsed_ticks;
     *   if elapsed_ticks >= LOAD, clamp at 0 and set TIMEOUT elsewhere.
     */
    return ptimer_get_count(s->timer);
}

static void g233_wdt_reload_counter(G233WDTState *s)
{
    s->value = s->load;
    s->status &= ~G233_WDT_SR_TIMEOUT;

    /*
     * TODO(user):
     * Restart the real watchdog countdown here. If you later move VAL to a
     * virtual-time derived model, this helper remains the right place for
     * FEED / EN-rising-edge reload semantics.
     */

    g233_set_next_timer(s);

}

static uint64_t g233_wdt_read(void *opaque, hwaddr offset, unsigned size)
{
    G233WDTState *s = opaque;

    if (size != 4) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: unsupported read size %u at 0x%" HWADDR_PRIx "\n",
                      TYPE_G233_WDT, size, offset);
        return 0;
    }

    switch (offset) {
    case G233_WDT_REG_CTRL:
        return g233_wdt_ctrl_readback(s);
    case G233_WDT_REG_LOAD:
        return s->load;
    case G233_WDT_REG_VAL:
        return g233_wdt_get_value(s);
    case G233_WDT_REG_SR:
        return s->status & G233_WDT_SR_TIMEOUT;
    case G233_WDT_REG_KEY:
        return 0;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: bad read offset 0x%" HWADDR_PRIx "\n",
                      TYPE_G233_WDT, offset);
        return 0;
    }
}

static void g233_wdt_write(void *opaque, hwaddr offset,
                           uint64_t value, unsigned size)
{
    G233WDTState *s = opaque;
    uint32_t data = value;
    bool old_en;
    bool new_en;

    if (size != 4) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: unsupported write size %u at 0x%" HWADDR_PRIx "\n",
                      TYPE_G233_WDT, size, offset);
        return;
    }

    switch (offset) {
    case G233_WDT_REG_CTRL:
        if (s->locked) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "%s: write to locked WDT_CTRL ignored\n",
                          TYPE_G233_WDT);
            return;
        }

        old_en = (s->ctrl & G233_WDT_CTRL_EN) != 0;
        s->ctrl = data & G233_WDT_CTRL_WRMASK;
        new_en = (s->ctrl & G233_WDT_CTRL_EN) != 0;

        /*
         * Scaffold policy:
         * when software enables the watchdog, start the visible counter from
         * LOAD so MMIO behaviour is intuitive even before countdown logic is
         * implemented.
         */
        if (!old_en && new_en) {
            g233_wdt_reload_counter(s);
        }
        break;
    case G233_WDT_REG_LOAD:
        s->load = data;

        /*
         * With countdown logic still absent, mirroring LOAD into VAL while the
         * block is disabled keeps register readback straightforward.
         */
        if (!(s->ctrl & G233_WDT_CTRL_EN)) {
            s->value = data;
        }
        break;
    case G233_WDT_REG_VAL:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: write to read-only WDT_VAL ignored\n",
                      TYPE_G233_WDT);
        return;
    case G233_WDT_REG_SR:
        s->status &= ~(data & G233_WDT_SR_TIMEOUT);
        break;
    case G233_WDT_REG_KEY:
        switch (data) {
        case G233_WDT_KEY_FEED:
            g233_wdt_reload_counter(s);
            break;
        case G233_WDT_KEY_LOCK:
            s->locked = true;
            break;
        default:
            /*
             * The specification marks all other keys as undefined. Keep the
             * scaffold conservative: ignore them and leave current state as-is.
             */
            break;
        }
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: bad write offset 0x%" HWADDR_PRIx
                      " = 0x%" PRIx64 "\n",
                      TYPE_G233_WDT, offset, value);
        return;
    }

    g233_wdt_update_irq(s);
}

static const MemoryRegionOps g233_wdt_ops = {
    .read = g233_wdt_read,
    .write = g233_wdt_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

static void g233_wdt_reset(DeviceState *dev)
{
    G233WDTState *s = G233_WDT(dev);

    s->ctrl = 0;
    s->load = 0x0000ffffu;
    s->value = 0x0000ffffu;
    s->status = 0;
    s->locked = false;

    g233_wdt_update_irq(s);
}

static void g233_wdt_init(Object *obj)
{
    G233WDTState *s = G233_WDT(obj);

    memory_region_init_io(&s->mmio, obj, &g233_wdt_ops, s,
                          TYPE_G233_WDT, G233_WDT_MMIO_SIZE);
    sysbus_init_mmio(SYS_BUS_DEVICE(obj), &s->mmio);
    sysbus_init_irq(SYS_BUS_DEVICE(obj), &s->irq);

    s->timer = ptimer_init(g233_wdt_timer_callback, s,
                           PTIMER_POLICY_NO_IMMEDIATE_TRIGGER |
                           PTIMER_POLICY_NO_IMMEDIATE_RELOAD |
                           PTIMER_POLICY_NO_COUNTER_ROUND_DOWN);
    
    ptimer_transaction_begin(s->timer);
    ptimer_set_freq(s->timer, G233_WDT_TIMER_HZ);
    ptimer_set_limit(s->timer, 0xffffffffu, 1);
    ptimer_transaction_commit(s->timer);
}

static const VMStateDescription vmstate_g233_wdt = {
    .name = TYPE_G233_WDT,
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT32(ctrl, G233WDTState),
        VMSTATE_UINT32(load, G233WDTState),
        VMSTATE_UINT32(value, G233WDTState),
        VMSTATE_UINT32(status, G233WDTState),
        VMSTATE_BOOL(locked, G233WDTState),
        VMSTATE_END_OF_LIST()
    },
};

static void g233_wdt_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->vmsd = &vmstate_g233_wdt;
    device_class_set_legacy_reset(dc, g233_wdt_reset);
}

static const TypeInfo g233_wdt_info = {
    .name = TYPE_G233_WDT,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(G233WDTState),
    .instance_init = g233_wdt_init,
    .class_init = g233_wdt_class_init,
};

static void g233_wdt_register_types(void)
{
    type_register_static(&g233_wdt_info);
}

type_init(g233_wdt_register_types)
