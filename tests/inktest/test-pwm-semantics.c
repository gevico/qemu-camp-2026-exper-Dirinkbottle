/*
 * Inktest: G233 PWM 控制器语义规格测试
 *
 * 这不是“为了当前实现去凑通过”的测试，而是一份可执行的规格说明：
 * 我们把 PWM 这个设备从软件可见角度拆成几条最基础语义，然后逐条验证。
 *
 * 当前测试只观察 MMIO 语义：
 * 1. 寄存器读写回显是否正确
 * 2. 使能后计数器是否随虚拟时间推进
 * 3. 禁用后计数器是否冻结
 * 4. 到达周期边界后 DONE 是否置位
 * 5. DONE 是否是 sticky 位，且支持 W1C 清除
 *
 * 这里不直接测试 pwm_out 引脚电平波形，因为 qtest 这份用例主要面向
 * “软件看见了什么寄存器行为”。如果之后你要验证真正输出波形，再补
 * 一份 line-level / gpio-level 的测试会更清楚。
 */

#include "qemu/osdep.h"
#include "../qtest/libqtest.h"

#define PWM_BASE             0x10015000ULL
#define PWM_GLB              (PWM_BASE + 0x00)

#define PWM_CH_CTRL(n)       (PWM_BASE + 0x10 + (n) * 0x10 + 0x00)
#define PWM_CH_PERIOD(n)     (PWM_BASE + 0x10 + (n) * 0x10 + 0x04)
#define PWM_CH_DUTY(n)       (PWM_BASE + 0x10 + (n) * 0x10 + 0x08)
#define PWM_CH_CNT(n)        (PWM_BASE + 0x10 + (n) * 0x10 + 0x0C)

#define PWM_GLB_CH_EN(n)     (1u << (n))
#define PWM_GLB_CH_DONE(n)   (1u << (4 + (n)))

#define PWM_CTRL_EN          (1u << 0)
#define PWM_CTRL_POL         (1u << 1)

/*
 * 目前设备默认时钟频率来自 g233_pwmctrl.c 的属性默认值：
 *   clock-frequency = 1_000_000 Hz
 *
 * 也就是：
 *   1 tick = 1 us = 1000 ns
 *
 * 这份测试直接按这个默认频率来推进虚拟时间。如果你后面改了设备默认时钟，
 * 就要同步更新这里的换算。
 */
#define PWM_CLOCK_HZ         1000000ULL
#define PWM_NS_PER_TICK      (1000000000ULL / PWM_CLOCK_HZ)

static QTestState *ink_pwm_init(void)
{
    return qtest_init("-machine g233 -m 2G");
}

static void ink_pwm_step_ticks(QTestState *qts, uint64_t ticks)
{
    qtest_clock_step(qts, ticks * PWM_NS_PER_TICK);
}

static void test_pwm_register_contract(void)
{
    QTestState *qts = ink_pwm_init();

    /*
     * 语义 1：
     * PERIOD / DUTY / CTRL 是软件配置寄存器，写进去的值应该能按原样读回来。
     * 这一步确认的是“编程接口存在”，不是时序行为。
     */
    qtest_writel(qts, PWM_CH_PERIOD(0), 100);
    qtest_writel(qts, PWM_CH_DUTY(0), 25);
    qtest_writel(qts, PWM_CH_CTRL(0), PWM_CTRL_POL);

    g_assert_cmpuint(qtest_readl(qts, PWM_CH_PERIOD(0)), ==, 100);
    g_assert_cmpuint(qtest_readl(qts, PWM_CH_DUTY(0)), ==, 25);
    g_assert_cmpuint(qtest_readl(qts, PWM_CH_CTRL(0)), ==, PWM_CTRL_POL);

    /*
     * 语义 2：
     * CHn_CTRL.EN 是每通道控制位；PWM_GLB.CHn_EN 是全局镜像位。
     * 打开 EN 后：
     * - CHn_CTRL 里能读到 EN
     * - PWM_GLB 对应镜像位也应该亮起来
     */
    qtest_writel(qts, PWM_CH_CTRL(0), PWM_CTRL_EN | PWM_CTRL_POL);

    g_assert_cmpuint(qtest_readl(qts, PWM_CH_CTRL(0)),
                     ==, PWM_CTRL_EN | PWM_CTRL_POL);
    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_EN(0),
                     ==, PWM_GLB_CH_EN(0));

    qtest_quit(qts);
}

static void test_pwm_counter_progress_and_freeze(void)
{
    QTestState *qts = ink_pwm_init();
    uint32_t cnt_before_disable;

    /*
     * 语义 3：
     * EN 从 0 -> 1 后，PWM 开始运行，CNT 应该从 0 起步。
     * 这里用较大的 period，避免太快 wrap，方便只观察“计数前进”。
     */
    qtest_writel(qts, PWM_CH_PERIOD(0), 1000);
    qtest_writel(qts, PWM_CH_DUTY(0), 250);
    qtest_writel(qts, PWM_CH_CTRL(0), PWM_CTRL_EN);

    g_assert_cmpuint(qtest_readl(qts, PWM_CH_CNT(0)), ==, 0);

    /*
     * 推进 37 个 tick。
     * 如果模型采用“按虚拟时间推导 CNT”的设计，那么现在应该读到 37。
     */
    ink_pwm_step_ticks(qts, 37);
    g_assert_cmpuint(qtest_readl(qts, PWM_CH_CNT(0)), ==, 37);

    /*
     * 语义 4：
     * EN 从 1 -> 0 后，计数器应停止继续前进。
     * 这里先记住禁用那一刻的 CNT，再推进时间，期望读回值不变。
     */
    qtest_writel(qts, PWM_CH_CTRL(0), 0);
    cnt_before_disable = qtest_readl(qts, PWM_CH_CNT(0));

    ink_pwm_step_ticks(qts, 200);
    g_assert_cmpuint(qtest_readl(qts, PWM_CH_CNT(0)), ==, cnt_before_disable);
    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_EN(0), ==, 0);

    qtest_quit(qts);
}

static void test_pwm_wrap_and_done_semantics(void)
{
    QTestState *qts = ink_pwm_init();

    /*
     * 语义 5：
     * CNT 到达一个 period 边界时，应发生一次“周期完成事件”。
     * 软件可见的最低要求通常是：
     * - DONE sticky 位置 1
     * - CNT 重新从 0 开始新一轮计数
     *
     * 这里 period = 100，所以推进 100 tick 后，应该正好跨过一次边界。
     */
    qtest_writel(qts, PWM_CH_PERIOD(0), 100);
    qtest_writel(qts, PWM_CH_DUTY(0), 30);
    qtest_writel(qts, PWM_CH_CTRL(0), PWM_CTRL_EN);

    ink_pwm_step_ticks(qts, 100);

    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_DONE(0),
                     ==, PWM_GLB_CH_DONE(0));
    g_assert_cmpuint(qtest_readl(qts, PWM_CH_CNT(0)), ==, 0);

    /*
     * 再推进 17 tick，确认 wrap 后确实是“新一轮从 0 重新计数”，
     * 而不是停在 period 边界不动。
     */
    ink_pwm_step_ticks(qts, 17);
    g_assert_cmpuint(qtest_readl(qts, PWM_CH_CNT(0)), ==, 17);

    qtest_quit(qts);
}

static void test_pwm_done_sticky_and_w1c(void)
{
    QTestState *qts = ink_pwm_init();

    /*
     * 语义 6：
     * DONE 是 sticky 状态位，不是瞬时脉冲。
     *
     * 一旦某个周期完成，它应该保持为 1，直到软件显式写 1 清除。
     * 这也是中断状态寄存器最常见的编程模型之一。
     */
    qtest_writel(qts, PWM_CH_PERIOD(0), 10);
    qtest_writel(qts, PWM_CH_DUTY(0), 5);
    qtest_writel(qts, PWM_CH_CTRL(0), PWM_CTRL_EN);

    ink_pwm_step_ticks(qts, 10);
    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_DONE(0),
                     ==, PWM_GLB_CH_DONE(0));

    /*
     * 再往后推进几个周期，如果 DONE 是 sticky，软件读到的状态仍应为 1。
     * 它不该因为硬件自己“过去很久了”就悄悄清零。
     */
    ink_pwm_step_ticks(qts, 30);
    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_DONE(0),
                     ==, PWM_GLB_CH_DONE(0));

    /*
     * 语义 7：
     * PWM_GLB 的 DONE 区域采用 W1C（write-one-to-clear）。
     * 即：
     * - 写 1 到对应 DONE 位：清零
     * - 写 0：不影响
     */
    qtest_writel(qts, PWM_GLB, PWM_GLB_CH_DONE(0));
    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_DONE(0), ==, 0);

    /*
     * 清掉后，如果硬件继续跑，再过一个完整 period，就应该再次置位。
     * 这能证明：
     * - 清零真的生效了
     * - 下一个周期完成事件还能重新打上 DONE
     */
    ink_pwm_step_ticks(qts, 10);
    g_assert_cmpuint(qtest_readl(qts, PWM_GLB) & PWM_GLB_CH_DONE(0),
                     ==, PWM_GLB_CH_DONE(0));

    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    g_test_add_func("/ink/pwm/register_contract", test_pwm_register_contract);
    g_test_add_func("/ink/pwm/counter_progress_and_freeze",
                    test_pwm_counter_progress_and_freeze);
    g_test_add_func("/ink/pwm/wrap_and_done_semantics",
                    test_pwm_wrap_and_done_semantics);
    g_test_add_func("/ink/pwm/done_sticky_and_w1c",
                    test_pwm_done_sticky_and_w1c);

    return g_test_run();
}
