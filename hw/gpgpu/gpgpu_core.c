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
#include "target/riscv/instmap.h"

#define GPGPU_CORE_MEM_SIZE_BYTE        1U
#define GPGPU_CORE_MEM_SIZE_HALF        2U
#define GPGPU_CORE_MEM_SIZE_WORD        4U

#define GPGPU_CORE_FFLAGS_MASK          0x1fU
#define GPGPU_CORE_FRM_SHIFT            5U
#define GPGPU_CORE_FRM_WIDTH            3U
#define GPGPU_CORE_FCSR_MASK            0xffU
#define GPGPU_CORE_FP32_SIGN_MASK       0x80000000U
#define GPGPU_CORE_FP32_MAG_MASK        0x7fffffffU
#define GPGPU_CORE_BF16_MASK            0xffffU
#define GPGPU_CORE_FP8_MASK             0xffU
#define GPGPU_CORE_FP4_MASK             0x0fU

#define GPGPU_CORE_FCLASS_NEG_INF       (1u << 0)
#define GPGPU_CORE_FCLASS_NEG_NORMAL    (1u << 1)
#define GPGPU_CORE_FCLASS_NEG_SUBNORMAL (1u << 2)
#define GPGPU_CORE_FCLASS_NEG_ZERO      (1u << 3)
#define GPGPU_CORE_FCLASS_POS_ZERO      (1u << 4)
#define GPGPU_CORE_FCLASS_POS_SUBNORMAL (1u << 5)
#define GPGPU_CORE_FCLASS_POS_NORMAL    (1u << 6)
#define GPGPU_CORE_FCLASS_POS_INF       (1u << 7)
#define GPGPU_CORE_FCLASS_SNAN          (1u << 8)
#define GPGPU_CORE_FCLASS_QNAN          (1u << 9)

#define GPGPU_RV_OPCODE_LOAD            0x03U
#define GPGPU_RV_OPCODE_LOAD_FP         0x07U
#define GPGPU_RV_OPCODE_OP_IMM          0x13U
#define GPGPU_RV_OPCODE_AUIPC           0x17U
#define GPGPU_RV_OPCODE_STORE           0x23U
#define GPGPU_RV_OPCODE_STORE_FP        0x27U
#define GPGPU_RV_OPCODE_OP              0x33U
#define GPGPU_RV_OPCODE_LUI             0x37U
#define GPGPU_RV_OPCODE_FMADD           0x43U
#define GPGPU_RV_OPCODE_FMSUB           0x47U
#define GPGPU_RV_OPCODE_FNMSUB          0x4bU
#define GPGPU_RV_OPCODE_FNMADD          0x4fU
#define GPGPU_RV_OPCODE_BRANCH          0x63U
#define GPGPU_RV_OPCODE_JALR            0x67U
#define GPGPU_RV_OPCODE_JAL             0x6fU
#define GPGPU_RV_OPCODE_SYSTEM          0x73U
#define GPGPU_RV_OPCODE_FP              0x53U

#define GPGPU_RV_FUNCT3_ADD_SUB         0x0U
#define GPGPU_RV_FUNCT3_SLL             0x1U
#define GPGPU_RV_FUNCT3_SLT             0x2U
#define GPGPU_RV_FUNCT3_SLTU            0x3U
#define GPGPU_RV_FUNCT3_XOR             0x4U
#define GPGPU_RV_FUNCT3_SR              0x5U
#define GPGPU_RV_FUNCT3_OR              0x6U
#define GPGPU_RV_FUNCT3_AND             0x7U

#define GPGPU_RV_FUNCT3_BEQ             0x0U
#define GPGPU_RV_FUNCT3_BNE             0x1U
#define GPGPU_RV_FUNCT3_BLT             0x4U
#define GPGPU_RV_FUNCT3_BGE             0x5U
#define GPGPU_RV_FUNCT3_BLTU            0x6U
#define GPGPU_RV_FUNCT3_BGEU            0x7U

#define GPGPU_RV_FUNCT3_LB              0x0U
#define GPGPU_RV_FUNCT3_LH              0x1U
#define GPGPU_RV_FUNCT3_LW              0x2U
#define GPGPU_RV_FUNCT3_LBU             0x4U
#define GPGPU_RV_FUNCT3_LHU             0x5U

#define GPGPU_RV_FUNCT3_SB              0x0U
#define GPGPU_RV_FUNCT3_SH              0x1U
#define GPGPU_RV_FUNCT3_SW              0x2U

#define GPGPU_RV_FUNCT3_CSRRW           0x1U
#define GPGPU_RV_FUNCT3_CSRRS           0x2U
#define GPGPU_RV_FUNCT3_CSRRC           0x3U
#define GPGPU_RV_FUNCT3_CSRRWI          0x5U
#define GPGPU_RV_FUNCT3_CSRRSI          0x6U
#define GPGPU_RV_FUNCT3_CSRRCI          0x7U

#define GPGPU_RV_FP_FUNCT3_WORD         0x2U
#define GPGPU_RV_FP_FUNCT3_FSGNJ        0x0U
#define GPGPU_RV_FP_FUNCT3_FSGNJN       0x1U
#define GPGPU_RV_FP_FUNCT3_FSGNJX       0x2U
#define GPGPU_RV_FP_FUNCT3_FMIN         0x0U
#define GPGPU_RV_FP_FUNCT3_FMAX         0x1U
#define GPGPU_RV_FP_FUNCT3_FLE          0x0U
#define GPGPU_RV_FP_FUNCT3_FLT          0x1U
#define GPGPU_RV_FP_FUNCT3_FEQ          0x2U
#define GPGPU_RV_FP_FUNCT3_FMV_X_W      0x0U
#define GPGPU_RV_FP_FUNCT3_FCLASS_S     0x1U
#define GPGPU_RV_FP_FUNCT3_FMV_W_X      0x0U

#define GPGPU_RV_FUNCT7_STD             0x00U
#define GPGPU_RV_FUNCT7_MUL             0x01U
#define GPGPU_RV_FUNCT7_SUB_SRA         0x20U

#define GPGPU_RV_FUNCT7_FADD_S          0x00U
#define GPGPU_RV_FUNCT7_FSUB_S          0x04U
#define GPGPU_RV_FUNCT7_FMUL_S          0x08U
#define GPGPU_RV_FUNCT7_FDIV_S          0x0cU
#define GPGPU_RV_FUNCT7_FSGNJ_S         0x10U
#define GPGPU_RV_FUNCT7_FMINMAX_S       0x14U
#define GPGPU_RV_FUNCT7_FCVT_BF16       0x22U
#define GPGPU_RV_FUNCT7_FCVT_FP8        0x24U
#define GPGPU_RV_FUNCT7_FCVT_E2M1       0x26U
#define GPGPU_RV_FUNCT7_FSQRT_S         0x2cU
#define GPGPU_RV_FUNCT7_FCMP_S          0x50U
#define GPGPU_RV_FUNCT7_FCVT_W_S        0x60U
#define GPGPU_RV_FUNCT7_FMV_X_W         0x70U
#define GPGPU_RV_FUNCT7_FCVT_S_W        0x68U
#define GPGPU_RV_FUNCT7_FMV_W_X         0x78U

#define GPGPU_RV_FP_RS2_FCVT_W          0x0U
#define GPGPU_RV_FP_RS2_FCVT_WU         0x1U
#define GPGPU_RV_FP_RS2_FSQRT           0x0U
#define GPGPU_RV_FP_RS2_FCVT_S_W        0x0U
#define GPGPU_RV_FP_RS2_FCVT_S_WU       0x1U
#define GPGPU_RV_FP_RS2_FMV             0x0U

#define GPGPU_RV_FP_RS2_S_BF16          0x0U
#define GPGPU_RV_FP_RS2_BF16_S          0x1U

#define GPGPU_RV_FP_RS2_S_E4M3          0x0U
#define GPGPU_RV_FP_RS2_E4M3_S          0x1U
#define GPGPU_RV_FP_RS2_S_E5M2          0x2U
#define GPGPU_RV_FP_RS2_E5M2_S          0x3U

#define GPGPU_RV_FP_RS2_S_E2M1          0x0U
#define GPGPU_RV_FP_RS2_E2M1_S          0x1U

#define GPGPU_RV_RM_RNE                 0x0U
#define GPGPU_RV_RM_RTZ                 0x1U
#define GPGPU_RV_RM_RDN                 0x2U
#define GPGPU_RV_RM_RUP                 0x3U
#define GPGPU_RV_RM_RMM                 0x4U
#define GPGPU_RV_RM_DYN                 0x7U

#define GPGPU_RV_FMA_FMT_SHIFT          25U
#define GPGPU_RV_FMA_FMT_WIDTH          2U
#define GPGPU_RV_FMT_S                  0x0U

#define GPGPU_RV_EBREAK_INST            0x00100073U
#define GPGPU_RV_PC_ALIGN_MASK          0x3U
#define GPGPU_RV_SHAMT_MASK             0x1fU
#define GPGPU_RV_U_IMM_MASK             0xfffff000U
#define GPGPU_RV_JALR_TARGET_MASK       (~1U)

typedef enum GPGPURVBranchInsn {
    GPGPU_RV_BRANCH_ILLEGAL = -1,
    GPGPU_RV_BRANCH_BEQ,
    GPGPU_RV_BRANCH_BNE,
    GPGPU_RV_BRANCH_BLT,
    GPGPU_RV_BRANCH_BGE,
    GPGPU_RV_BRANCH_BLTU,
    GPGPU_RV_BRANCH_BGEU,
} GPGPURVBranchInsn;

typedef enum GPGPURVLoadInsn {
    GPGPU_RV_LOAD_ILLEGAL = -1,
    GPGPU_RV_LOAD_LB,
    GPGPU_RV_LOAD_LH,
    GPGPU_RV_LOAD_LW,
    GPGPU_RV_LOAD_LBU,
    GPGPU_RV_LOAD_LHU,
} GPGPURVLoadInsn;

typedef enum GPGPURVStoreInsn {
    GPGPU_RV_STORE_ILLEGAL = -1,
    GPGPU_RV_STORE_SB,
    GPGPU_RV_STORE_SH,
    GPGPU_RV_STORE_SW,
} GPGPURVStoreInsn;

typedef enum GPGPURVOpImmInsn {
    GPGPU_RV_OP_IMM_ILLEGAL = -1,
    GPGPU_RV_OP_IMM_ADDI,
    GPGPU_RV_OP_IMM_SLLI,
    GPGPU_RV_OP_IMM_SLTI,
    GPGPU_RV_OP_IMM_SLTIU,
    GPGPU_RV_OP_IMM_XORI,
    GPGPU_RV_OP_IMM_SRLI,
    GPGPU_RV_OP_IMM_SRAI,
    GPGPU_RV_OP_IMM_ORI,
    GPGPU_RV_OP_IMM_ANDI,
} GPGPURVOpImmInsn;

typedef enum GPGPURVOpInsn {
    GPGPU_RV_OP_ILLEGAL = -1,
    GPGPU_RV_OP_ADD,
    GPGPU_RV_OP_SUB,
    GPGPU_RV_OP_SLL,
    GPGPU_RV_OP_SLT,
    GPGPU_RV_OP_SLTU,
    GPGPU_RV_OP_XOR,
    GPGPU_RV_OP_SRL,
    GPGPU_RV_OP_SRA,
    GPGPU_RV_OP_OR,
    GPGPU_RV_OP_AND,
    GPGPU_RV_OP_MUL,
} GPGPURVOpInsn;

typedef enum GPGPURVSystemInsn {
    GPGPU_RV_SYSTEM_ILLEGAL = -1,
    GPGPU_RV_SYSTEM_EBREAK,
    GPGPU_RV_SYSTEM_CSRRW,
    GPGPU_RV_SYSTEM_CSRRS,
    GPGPU_RV_SYSTEM_CSRRC,
    GPGPU_RV_SYSTEM_CSRRWI,
    GPGPU_RV_SYSTEM_CSRRSI,
    GPGPU_RV_SYSTEM_CSRRCI,
} GPGPURVSystemInsn;

static GPGPURVBranchInsn gpgpu_core_decode_branch_insn(uint32_t funct3)
{
    switch (funct3) {
    case GPGPU_RV_FUNCT3_BEQ:
        return GPGPU_RV_BRANCH_BEQ;
    case GPGPU_RV_FUNCT3_BNE:
        return GPGPU_RV_BRANCH_BNE;
    case GPGPU_RV_FUNCT3_BLT:
        return GPGPU_RV_BRANCH_BLT;
    case GPGPU_RV_FUNCT3_BGE:
        return GPGPU_RV_BRANCH_BGE;
    case GPGPU_RV_FUNCT3_BLTU:
        return GPGPU_RV_BRANCH_BLTU;
    case GPGPU_RV_FUNCT3_BGEU:
        return GPGPU_RV_BRANCH_BGEU;
    default:
        return GPGPU_RV_BRANCH_ILLEGAL;
    }
}

static GPGPURVLoadInsn gpgpu_core_decode_load_insn(uint32_t funct3)
{
    switch (funct3) {
    case GPGPU_RV_FUNCT3_LB:
        return GPGPU_RV_LOAD_LB;
    case GPGPU_RV_FUNCT3_LH:
        return GPGPU_RV_LOAD_LH;
    case GPGPU_RV_FUNCT3_LW:
        return GPGPU_RV_LOAD_LW;
    case GPGPU_RV_FUNCT3_LBU:
        return GPGPU_RV_LOAD_LBU;
    case GPGPU_RV_FUNCT3_LHU:
        return GPGPU_RV_LOAD_LHU;
    default:
        return GPGPU_RV_LOAD_ILLEGAL;
    }
}

static GPGPURVStoreInsn gpgpu_core_decode_store_insn(uint32_t funct3)
{
    switch (funct3) {
    case GPGPU_RV_FUNCT3_SB:
        return GPGPU_RV_STORE_SB;
    case GPGPU_RV_FUNCT3_SH:
        return GPGPU_RV_STORE_SH;
    case GPGPU_RV_FUNCT3_SW:
        return GPGPU_RV_STORE_SW;
    default:
        return GPGPU_RV_STORE_ILLEGAL;
    }
}

static GPGPURVOpImmInsn gpgpu_core_decode_op_imm_insn(uint32_t funct3,
                                                      uint32_t funct7)
{
    switch (funct3) {
    case GPGPU_RV_FUNCT3_ADD_SUB:
        return GPGPU_RV_OP_IMM_ADDI;
    case GPGPU_RV_FUNCT3_SLL:
        return funct7 == GPGPU_RV_FUNCT7_STD ?
               GPGPU_RV_OP_IMM_SLLI : GPGPU_RV_OP_IMM_ILLEGAL;
    case GPGPU_RV_FUNCT3_SLT:
        return GPGPU_RV_OP_IMM_SLTI;
    case GPGPU_RV_FUNCT3_SLTU:
        return GPGPU_RV_OP_IMM_SLTIU;
    case GPGPU_RV_FUNCT3_XOR:
        return GPGPU_RV_OP_IMM_XORI;
    case GPGPU_RV_FUNCT3_SR:
        switch (funct7) {
        case GPGPU_RV_FUNCT7_STD:
            return GPGPU_RV_OP_IMM_SRLI;
        case GPGPU_RV_FUNCT7_SUB_SRA:
            return GPGPU_RV_OP_IMM_SRAI;
        default:
            return GPGPU_RV_OP_IMM_ILLEGAL;
        }
    case GPGPU_RV_FUNCT3_OR:
        return GPGPU_RV_OP_IMM_ORI;
    case GPGPU_RV_FUNCT3_AND:
        return GPGPU_RV_OP_IMM_ANDI;
    default:
        return GPGPU_RV_OP_IMM_ILLEGAL;
    }
}

static GPGPURVOpInsn gpgpu_core_decode_op_insn(uint32_t funct3,
                                               uint32_t funct7)
{
    switch (funct7) {
    case GPGPU_RV_FUNCT7_STD:
        switch (funct3) {
        case GPGPU_RV_FUNCT3_ADD_SUB:
            return GPGPU_RV_OP_ADD;
        case GPGPU_RV_FUNCT3_SLL:
            return GPGPU_RV_OP_SLL;
        case GPGPU_RV_FUNCT3_SLT:
            return GPGPU_RV_OP_SLT;
        case GPGPU_RV_FUNCT3_SLTU:
            return GPGPU_RV_OP_SLTU;
        case GPGPU_RV_FUNCT3_XOR:
            return GPGPU_RV_OP_XOR;
        case GPGPU_RV_FUNCT3_SR:
            return GPGPU_RV_OP_SRL;
        case GPGPU_RV_FUNCT3_OR:
            return GPGPU_RV_OP_OR;
        case GPGPU_RV_FUNCT3_AND:
            return GPGPU_RV_OP_AND;
        default:
            return GPGPU_RV_OP_ILLEGAL;
        }
    case GPGPU_RV_FUNCT7_SUB_SRA:
        switch (funct3) {
        case GPGPU_RV_FUNCT3_ADD_SUB:
            return GPGPU_RV_OP_SUB;
        case GPGPU_RV_FUNCT3_SR:
            return GPGPU_RV_OP_SRA;
        default:
            return GPGPU_RV_OP_ILLEGAL;
        }
    case GPGPU_RV_FUNCT7_MUL:
        return funct3 == GPGPU_RV_FUNCT3_ADD_SUB ?
               GPGPU_RV_OP_MUL : GPGPU_RV_OP_ILLEGAL;
    default:
        return GPGPU_RV_OP_ILLEGAL;
    }
}

static GPGPURVSystemInsn gpgpu_core_decode_system_insn(uint32_t inst,
                                                       uint32_t funct3)
{
    if (inst == GPGPU_RV_EBREAK_INST) {
        return GPGPU_RV_SYSTEM_EBREAK;
    }

    switch (funct3) {
    case GPGPU_RV_FUNCT3_CSRRW:
        return GPGPU_RV_SYSTEM_CSRRW;
    case GPGPU_RV_FUNCT3_CSRRS:
        return GPGPU_RV_SYSTEM_CSRRS;
    case GPGPU_RV_FUNCT3_CSRRC:
        return GPGPU_RV_SYSTEM_CSRRC;
    case GPGPU_RV_FUNCT3_CSRRWI:
        return GPGPU_RV_SYSTEM_CSRRWI;
    case GPGPU_RV_FUNCT3_CSRRSI:
        return GPGPU_RV_SYSTEM_CSRRSI;
    case GPGPU_RV_FUNCT3_CSRRCI:
        return GPGPU_RV_SYSTEM_CSRRCI;
    default:
        return GPGPU_RV_SYSTEM_ILLEGAL;
    }
}

static void gpgpu_core_report_mem_fault(GPGPUState *s, uint32_t addr,
                                        unsigned size, const char *op)
{
    s->global_status |= GPGPU_STATUS_ERROR;
    s->error_status |= GPGPU_ERR_VRAM_FAULT;
    s->irq_status |= GPGPU_IRQ_ERROR;

    qemu_log_mask(LOG_GUEST_ERROR,
                  "gpgpu: %s out of range addr=0x%08x size=%u vram_size=0x%"
                  PRIx64 "\n",
                  op, addr, size, s->vram_size);
}

static void gpgpu_core_set_lane_context(GPGPUState *s, const GPGPUWarp *warp,
                                        uint32_t lane_idx)
{
    uint32_t linear_tid = warp->thread_id_base + lane_idx;
    uint64_t plane_size = (uint64_t)s->kernel.block_dim[0] *
                          s->kernel.block_dim[1];
    uint32_t plane_rem = linear_tid;

    if (plane_size) {
        s->simt.thread_id[2] = linear_tid / plane_size;
        plane_rem = linear_tid % plane_size;
    } else {
        s->simt.thread_id[2] = 0;
    }

    if (s->kernel.block_dim[0]) {
        s->simt.thread_id[1] = plane_rem / s->kernel.block_dim[0];
        s->simt.thread_id[0] = plane_rem % s->kernel.block_dim[0];
    } else {
        s->simt.thread_id[0] = 0;
        s->simt.thread_id[1] = 0;
    }

    s->simt.block_id[0] = warp->block_id[0];
    s->simt.block_id[1] = warp->block_id[1];
    s->simt.block_id[2] = warp->block_id[2];
    s->simt.warp_id = warp->warp_id;
    s->simt.lane_id = lane_idx;
    s->simt.thread_mask = warp->active_mask;
}

static int gpgpu_core_ctrl_read(GPGPUState *s, uint32_t addr, uint32_t *value)
{
    switch (addr) {
    case GPGPU_CORE_CTRL_THREAD_ID_X:
        *value = s->simt.thread_id[0];
        return 0;
    case GPGPU_CORE_CTRL_THREAD_ID_Y:
        *value = s->simt.thread_id[1];
        return 0;
    case GPGPU_CORE_CTRL_THREAD_ID_Z:
        *value = s->simt.thread_id[2];
        return 0;
    case GPGPU_CORE_CTRL_BLOCK_ID_X:
        *value = s->simt.block_id[0];
        return 0;
    case GPGPU_CORE_CTRL_BLOCK_ID_Y:
        *value = s->simt.block_id[1];
        return 0;
    case GPGPU_CORE_CTRL_BLOCK_ID_Z:
        *value = s->simt.block_id[2];
        return 0;
    case GPGPU_CORE_CTRL_BLOCK_DIM_X:
        *value = s->kernel.block_dim[0];
        return 0;
    case GPGPU_CORE_CTRL_BLOCK_DIM_Y:
        *value = s->kernel.block_dim[1];
        return 0;
    case GPGPU_CORE_CTRL_BLOCK_DIM_Z:
        *value = s->kernel.block_dim[2];
        return 0;
    case GPGPU_CORE_CTRL_GRID_DIM_X:
        *value = s->kernel.grid_dim[0];
        return 0;
    case GPGPU_CORE_CTRL_GRID_DIM_Y:
        *value = s->kernel.grid_dim[1];
        return 0;
    case GPGPU_CORE_CTRL_GRID_DIM_Z:
        *value = s->kernel.grid_dim[2];
        return 0;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: unsupported ctrl read addr=0x%08x\n", addr);
        return -1;
    }
}

static int gpgpu_core_ctrl_write(GPGPUState *s, uint32_t addr, uint32_t value)
{
    switch (addr) {
    case GPGPU_CORE_BARRIER:
        s->simt.barrier_count++;
        if (s->simt.barrier_count >= s->simt.barrier_target) {
            s->simt.barrier_count = 0;
            s->simt.barrier_active = false;
        } else {
            s->simt.barrier_active = true;
        }
        return 0;
    case GPGPU_CORE_THREAD_MASK:
        s->simt.thread_mask = value;
        return 0;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: unsupported ctrl write addr=0x%08x val=0x%08x\n",
                      addr, value);
        return -1;
    }
}

static int gpgpu_core_mem_read(GPGPUState *s, uint32_t addr, unsigned size,
                               uint32_t *value)
{
    uint64_t end = (uint64_t)addr + size;
    uint32_t data = 0;
    unsigned i;

    if (addr >= GPGPU_CORE_CTRL_BASE) {
        if (size != GPGPU_CORE_MEM_SIZE_WORD) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: ctrl read requires 32-bit access addr=0x%08x "
                          "size=%u\n", addr, size);
            return -1;
        }
        return gpgpu_core_ctrl_read(s, addr, value);
    }

    if (!s->vram_ptr || !size || end > s->vram_size) {
        gpgpu_core_report_mem_fault(s, addr, size, "read");
        return -1;
    }

    for (i = 0; i < size; i++) {
        data |= (uint32_t)s->vram_ptr[addr + i] << (i * 8);
    }

    *value = data;
    return 0;
}

static int gpgpu_core_mem_write(GPGPUState *s, uint32_t addr, uint32_t value,
                                unsigned size)
{
    uint64_t end = (uint64_t)addr + size;
    unsigned i;

    if (addr >= GPGPU_CORE_CTRL_BASE) {
        if (size != GPGPU_CORE_MEM_SIZE_WORD) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: ctrl write requires 32-bit access addr=0x%08x "
                          "size=%u\n", addr, size);
            return -1;
        }
        return gpgpu_core_ctrl_write(s, addr, value);
    }

    if (!s->vram_ptr || !size || end > s->vram_size) {
        gpgpu_core_report_mem_fault(s, addr, size, "write");
        return -1;
    }

    for (i = 0; i < size; i++) {
        s->vram_ptr[addr + i] = extract32(value, i * 8, 8);
    }

    return 0;
}

static bool gpgpu_core_map_softfloat_rm(uint32_t rm, FloatRoundMode *soft_rm)
{
    switch (rm) {
    case GPGPU_RV_RM_RNE:
        *soft_rm = float_round_nearest_even;
        return true;
    case GPGPU_RV_RM_RTZ:
        *soft_rm = float_round_to_zero;
        return true;
    case GPGPU_RV_RM_RDN:
        *soft_rm = float_round_down;
        return true;
    case GPGPU_RV_RM_RUP:
        *soft_rm = float_round_up;
        return true;
    case GPGPU_RV_RM_RMM:
        *soft_rm = float_round_ties_away;
        return true;
    default:
        return false;
    }
}

static void gpgpu_core_sync_fp_status_from_fcsr(GPGPULane *lane)
{
    uint32_t rm = extract32(lane->fcsr, GPGPU_CORE_FRM_SHIFT,
                            GPGPU_CORE_FRM_WIDTH);
    FloatRoundMode soft_rm;

    set_float_exception_flags(lane->fcsr & GPGPU_CORE_FFLAGS_MASK,
                              &lane->fp_status);

    if (gpgpu_core_map_softfloat_rm(rm, &soft_rm)) {
        set_float_rounding_mode(soft_rm, &lane->fp_status);
    }
}

static void gpgpu_core_sync_fcsr_from_fp_status(GPGPULane *lane)
{
    lane->fcsr = deposit32(lane->fcsr, 0, 5,
                           get_float_exception_flags(&lane->fp_status) &
                           GPGPU_CORE_FFLAGS_MASK);
}

static int gpgpu_core_resolve_fp_rm(const GPGPULane *lane, uint32_t rm_field,
                                    uint32_t *rm_effective,
                                    const char **rm_name);

static int gpgpu_core_prepare_fp_op(GPGPULane *lane, bool has_rm,
                                    uint32_t rm_field)
{
    uint32_t rm_effective;
    const char *rm_name;
    FloatRoundMode soft_rm;

    gpgpu_core_sync_fp_status_from_fcsr(lane);

    if (!has_rm) {
        return 0;
    }

    if (gpgpu_core_resolve_fp_rm(lane, rm_field, &rm_effective, &rm_name) < 0) {
        return -1;
    }
    (void)rm_name;

    if (!gpgpu_core_map_softfloat_rm(rm_effective, &soft_rm)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: failed to map fp rm field=0x%x effective=0x%x\n",
                      rm_field, rm_effective);
        return -1;
    }

    set_float_rounding_mode(soft_rm, &lane->fp_status);
    return 0;
}

static inline float32 gpgpu_core_get_fpr_s(const GPGPULane *lane, uint32_t reg)
{
    return make_float32(lane->fpr[reg]);
}

static inline void gpgpu_core_set_fpr_s(GPGPULane *lane, uint32_t reg,
                                        float32 value)
{
    lane->fpr[reg] = float32_val(value);
}

static uint32_t gpgpu_core_fclass_s(float32 value)
{
    bool sign = float32_is_neg(value);

    if (float32_is_infinity(value)) {
        return sign ? GPGPU_CORE_FCLASS_NEG_INF : GPGPU_CORE_FCLASS_POS_INF;
    }

    if (float32_is_zero(value)) {
        return sign ? GPGPU_CORE_FCLASS_NEG_ZERO : GPGPU_CORE_FCLASS_POS_ZERO;
    }

    if (float32_is_zero_or_denormal(value)) {
        return sign ? GPGPU_CORE_FCLASS_NEG_SUBNORMAL :
                      GPGPU_CORE_FCLASS_POS_SUBNORMAL;
    }

    if (float32_is_any_nan(value)) {
        float_status nan_status = { };

        return float32_is_quiet_nan(value, &nan_status) ?
               GPGPU_CORE_FCLASS_QNAN : GPGPU_CORE_FCLASS_SNAN;
    }

    return sign ? GPGPU_CORE_FCLASS_NEG_NORMAL :
                  GPGPU_CORE_FCLASS_POS_NORMAL;
}

static float4_e2m1 gpgpu_core_float32_to_e2m1(float32 value)
{
    static const uint32_t e2m1_mid_0p25 = 0x3e800000U;
    static const uint32_t e2m1_mid_0p75 = 0x3f400000U;
    static const uint32_t e2m1_mid_1p25 = 0x3fa00000U;
    static const uint32_t e2m1_mid_1p75 = 0x3fe00000U;
    static const uint32_t e2m1_mid_2p50 = 0x40200000U;
    static const uint32_t e2m1_mid_3p50 = 0x40600000U;
    static const uint32_t e2m1_mid_5p00 = 0x40a00000U;
    /*
     * TODO(user): 如果你想真正吃透 FP4，建议把这个“阈值量化器”替换成
     * 自己推导的 sign/exponent/mantissa 打包逻辑。当前实现先保证语义和
     * 测试值可用：最近值量化，超范围饱和到 +/-6.0。
     */
    float32 abs_value = float32_abs(value);
    bool sign = float32_is_neg(value);
    uint32_t raw = float32_val(abs_value);
    float4_e2m1 code;

    if (float32_is_any_nan(value) || float32_is_infinity(value)) {
        return sign ? 0xf : 0x7;
    }

    if (raw < e2m1_mid_0p25) {
        code = 0x0;
    } else if (raw < e2m1_mid_0p75) {
        code = 0x1;
    } else if (raw < e2m1_mid_1p25) {
        code = 0x2;
    } else if (raw < e2m1_mid_1p75) {
        code = 0x3;
    } else if (raw < e2m1_mid_2p50) {
        code = 0x4;
    } else if (raw < e2m1_mid_3p50) {
        code = 0x5;
    } else if (raw < e2m1_mid_5p00) {
        code = 0x6;
    } else {
        code = 0x7;
    }

    return code | (sign ? 0x8 : 0x0);
}

static float32 gpgpu_core_e2m1_to_float32(float4_e2m1 value,
                                          float_status *status)
{
    float8_e4m3 fp8 = float4_e2m1_to_float8_e4m3(value & GPGPU_CORE_FP4_MASK,
                                                 status);
    bfloat16 bf16 = float8_e4m3_to_bfloat16(fp8, status);

    return bfloat16_to_float32(bf16, status);
}

static int gpgpu_core_read_csr(const GPGPULane *lane, uint32_t csr,
                               uint32_t *value)
{
    switch (csr) {
    case CSR_MHARTID:
        *value = lane->mhartid;
        return 0;
    case CSR_FFLAGS:
        *value = lane->fcsr & GPGPU_CORE_FFLAGS_MASK;
        return 0;
    case CSR_FRM:
        *value = extract32(lane->fcsr, GPGPU_CORE_FRM_SHIFT,
                           GPGPU_CORE_FRM_WIDTH);
        return 0;
    case CSR_FCSR:
        *value = lane->fcsr & GPGPU_CORE_FCSR_MASK;
        return 0;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: unsupported csr read csr=0x%03x\n", csr);
        return -1;
    }
}

static int gpgpu_core_write_csr(GPGPULane *lane, uint32_t csr, uint32_t value)
{
    switch (csr) {
    case CSR_FFLAGS:
        lane->fcsr = deposit32(lane->fcsr, 0, 5, value);
        break;
    case CSR_FRM:
        lane->fcsr = deposit32(lane->fcsr, GPGPU_CORE_FRM_SHIFT,
                               GPGPU_CORE_FRM_WIDTH, value);
        break;
    case CSR_FCSR:
        lane->fcsr = value & GPGPU_CORE_FCSR_MASK;
        break;
    case CSR_MHARTID:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: mhartid is read-only\n");
        return -1;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: unsupported csr write csr=0x%03x val=0x%08x\n",
                      csr, value);
        return -1;
    }

    gpgpu_core_sync_fp_status_from_fcsr(lane);
    return 0;
}

static const char *gpgpu_core_fp_rm_name(uint32_t rm)
{
    switch (rm) {
    case GPGPU_RV_RM_RNE:
        return "RNE";
    case GPGPU_RV_RM_RTZ:
        return "RTZ";
    case GPGPU_RV_RM_RDN:
        return "RDN";
    case GPGPU_RV_RM_RUP:
        return "RUP";
    case GPGPU_RV_RM_RMM:
        return "RMM";
    default:
        return "ILLEGAL";
    }
}

static int gpgpu_core_resolve_fp_rm(const GPGPULane *lane, uint32_t rm_field,
                                    uint32_t *rm_effective,
                                    const char **rm_name)
{
    uint32_t effective_rm = rm_field;

    if (rm_field == GPGPU_RV_RM_DYN) {
        effective_rm = extract32(lane->fcsr, GPGPU_CORE_FRM_SHIFT,
                                 GPGPU_CORE_FRM_WIDTH);
    }

    switch (effective_rm) {
    case GPGPU_RV_RM_RNE:
    case GPGPU_RV_RM_RTZ:
    case GPGPU_RV_RM_RDN:
    case GPGPU_RV_RM_RUP:
    // round to far zero
    case GPGPU_RV_RM_RMM:
        *rm_effective = effective_rm;
        *rm_name = gpgpu_core_fp_rm_name(effective_rm);
        return 0;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: illegal fp rm field=0x%x effective=0x%x\n",
                      rm_field, effective_rm);
        return -1;
    }
}

static int gpgpu_core_exec_fp_load_store(GPGPUState *s, GPGPULane *lane,
                                         uint32_t inst, uint32_t pc,
                                         uint32_t opcode, uint32_t funct3,
                                         uint32_t rd, uint32_t rs1_val,
                                         uint32_t rs2)
{
    uint32_t addr;
    uint32_t mem_val;

    if (funct3 != GPGPU_RV_FP_FUNCT3_WORD) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: unsupported fp load/store funct3=%u "
                      "inst=0x%08x pc=0x%08x\n",
                      funct3, inst, pc);
        return -1;
    }

    if (opcode == GPGPU_RV_OPCODE_LOAD_FP) {
        addr = rs1_val + sextract32(inst, 20, 12);
        if (gpgpu_core_mem_read(s, addr, GPGPU_CORE_MEM_SIZE_WORD, &mem_val) < 0) {
            return -1;
        }
        lane->fpr[rd] = mem_val;
        return 0;
    }

    if (opcode == GPGPU_RV_OPCODE_STORE_FP) {
        addr = rs1_val + (int32_t)GET_STORE_IMM(inst);
        return gpgpu_core_mem_write(s, addr, lane->fpr[rs2],
                                    GPGPU_CORE_MEM_SIZE_WORD);
    }

    qemu_log_mask(LOG_GUEST_ERROR,
                  "gpgpu: unsupported fp load/store opcode=0x%02x "
                  "inst=0x%08x pc=0x%08x\n",
                  opcode, inst, pc);
    return -1;
}

static int gpgpu_core_exec_fp_fma(GPGPULane *lane, uint32_t inst,
                                  uint32_t pc, uint32_t opcode, uint32_t rd,
                                  uint32_t rs1, uint32_t rs2)
{
    uint32_t fmt = extract32(inst, GPGPU_RV_FMA_FMT_SHIFT,
                             GPGPU_RV_FMA_FMT_WIDTH);
    uint32_t rs3 = GET_RS3(inst);
    uint32_t rm_field = GET_RM(inst);
    int muladd_flags = 0;
    float32 result;

    if (fmt != GPGPU_RV_FMT_S) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: unsupported non-single FMA fmt=%u "
                      "inst=0x%08x pc=0x%08x\n",
                      fmt, inst, pc);
        return -1;
    }

    switch (opcode) {
    case GPGPU_RV_OPCODE_FMADD:
        break;
    case GPGPU_RV_OPCODE_FMSUB:
        muladd_flags = float_muladd_negate_c;
        break;
    case GPGPU_RV_OPCODE_FNMSUB:
        muladd_flags = float_muladd_negate_product;
        break;
    case GPGPU_RV_OPCODE_FNMADD:
        muladd_flags = float_muladd_negate_product |
                       float_muladd_negate_c;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: unsupported FMA opcode=0x%02x "
                      "inst=0x%08x pc=0x%08x\n",
                      opcode, inst, pc);
        return -1;
    }

    if (gpgpu_core_prepare_fp_op(lane, true, rm_field) < 0) {
        return -1;
    }

    result = float32_muladd(gpgpu_core_get_fpr_s(lane, rs1),
                            gpgpu_core_get_fpr_s(lane, rs2),
                            gpgpu_core_get_fpr_s(lane, rs3),
                            muladd_flags, &lane->fp_status);
    gpgpu_core_set_fpr_s(lane, rd, result);
    gpgpu_core_sync_fcsr_from_fp_status(lane);
    return 0;
}

static int gpgpu_core_exec_op_fp(GPGPULane *lane, uint32_t inst,
                                 uint32_t pc, uint32_t rd, uint32_t rs1,
                                 uint32_t rs2, uint32_t funct3,
                                 uint32_t funct7)
{
    uint32_t rm_field = GET_RM(inst);
    float32 frs1 = gpgpu_core_get_fpr_s(lane, rs1);
    float32 frs2 = gpgpu_core_get_fpr_s(lane, rs2);

    switch (funct7) {
    case GPGPU_RV_FUNCT7_FADD_S:
        if (gpgpu_core_prepare_fp_op(lane, true, rm_field) < 0) {
            return -1;
        }
        gpgpu_core_set_fpr_s(lane, rd,
                             float32_add(frs1, frs2, &lane->fp_status));
        gpgpu_core_sync_fcsr_from_fp_status(lane);
        return 0;
    case GPGPU_RV_FUNCT7_FSUB_S:
        if (gpgpu_core_prepare_fp_op(lane, true, rm_field) < 0) {
            return -1;
        }
        gpgpu_core_set_fpr_s(lane, rd,
                             float32_sub(frs1, frs2, &lane->fp_status));
        gpgpu_core_sync_fcsr_from_fp_status(lane);
        return 0;
    case GPGPU_RV_FUNCT7_FMUL_S:
        if (gpgpu_core_prepare_fp_op(lane, true, rm_field) < 0) {
            return -1;
        }
        gpgpu_core_set_fpr_s(lane, rd,
                             float32_mul(frs1, frs2, &lane->fp_status));
        gpgpu_core_sync_fcsr_from_fp_status(lane);
        return 0;
    case GPGPU_RV_FUNCT7_FDIV_S:
        if (gpgpu_core_prepare_fp_op(lane, true, rm_field) < 0) {
            return -1;
        }
        gpgpu_core_set_fpr_s(lane, rd,
                             float32_div(frs1, frs2, &lane->fp_status));
        gpgpu_core_sync_fcsr_from_fp_status(lane);
        return 0;
    case GPGPU_RV_FUNCT7_FSGNJ_S:
        switch (funct3) {
        case GPGPU_RV_FP_FUNCT3_FSGNJ:
            lane->fpr[rd] = (lane->fpr[rs1] & GPGPU_CORE_FP32_MAG_MASK) |
                            (lane->fpr[rs2] & GPGPU_CORE_FP32_SIGN_MASK);
            return 0;
        case GPGPU_RV_FP_FUNCT3_FSGNJN:
            lane->fpr[rd] = (lane->fpr[rs1] & GPGPU_CORE_FP32_MAG_MASK) |
                            ((~lane->fpr[rs2]) & GPGPU_CORE_FP32_SIGN_MASK);
            return 0;
        case GPGPU_RV_FP_FUNCT3_FSGNJX:
            lane->fpr[rd] = lane->fpr[rs1] ^
                            (lane->fpr[rs2] & GPGPU_CORE_FP32_SIGN_MASK);
            return 0;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: unsupported FSGNJ.S funct3=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          funct3, inst, pc);
            return -1;
        }
    case GPGPU_RV_FUNCT7_FMINMAX_S:
        switch (funct3) {
        case GPGPU_RV_FP_FUNCT3_FMIN:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            gpgpu_core_set_fpr_s(lane, rd,
                                 float32_minimum_number(frs1, frs2,
                                                        &lane->fp_status));
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        case GPGPU_RV_FP_FUNCT3_FMAX:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            gpgpu_core_set_fpr_s(lane, rd,
                                 float32_maximum_number(frs1, frs2,
                                                        &lane->fp_status));
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: unsupported FMIN/FMAX.S funct3=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          funct3, inst, pc);
            return -1;
        }
    case GPGPU_RV_FUNCT7_FSQRT_S:
        if (rs2 != GPGPU_RV_FP_RS2_FSQRT) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: illegal FSQRT.S rs2=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          rs2, inst, pc);
            return -1;
        }
        if (gpgpu_core_prepare_fp_op(lane, true, rm_field) < 0) {
            return -1;
        }
        gpgpu_core_set_fpr_s(lane, rd,
                             float32_sqrt(frs1, &lane->fp_status));
        gpgpu_core_sync_fcsr_from_fp_status(lane);
        return 0;
    case GPGPU_RV_FUNCT7_FCMP_S:
        switch (funct3) {
        case GPGPU_RV_FP_FUNCT3_FEQ:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            lane->gpr[rd] = float32_eq_quiet(frs1, frs2, &lane->fp_status);
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        case GPGPU_RV_FP_FUNCT3_FLT:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            lane->gpr[rd] = float32_lt(frs1, frs2, &lane->fp_status);
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        case GPGPU_RV_FP_FUNCT3_FLE:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            lane->gpr[rd] = float32_le(frs1, frs2, &lane->fp_status);
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: unsupported FCMP.S funct3=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          funct3, inst, pc);
            return -1;
        }
    case GPGPU_RV_FUNCT7_FCVT_W_S:
        switch (rs2) {
        case GPGPU_RV_FP_RS2_FCVT_W:
            if (gpgpu_core_prepare_fp_op(lane, true, rm_field) < 0) {
                return -1;
            }
            lane->gpr[rd] = float32_to_int32(frs1, &lane->fp_status);
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        case GPGPU_RV_FP_RS2_FCVT_WU:
            if (gpgpu_core_prepare_fp_op(lane, true, rm_field) < 0) {
                return -1;
            }
            lane->gpr[rd] = float32_to_uint32(frs1, &lane->fp_status);
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: unsupported FCVT.*.S rs2=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          rs2, inst, pc);
            return -1;
        }
    case GPGPU_RV_FUNCT7_FCVT_S_W:
        switch (rs2) {
        case GPGPU_RV_FP_RS2_FCVT_S_W:
            if (gpgpu_core_prepare_fp_op(lane, true, rm_field) < 0) {
                return -1;
            }
            gpgpu_core_set_fpr_s(lane, rd,
                                 int32_to_float32((int32_t)lane->gpr[rs1],
                                                  &lane->fp_status));
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        case GPGPU_RV_FP_RS2_FCVT_S_WU:
            if (gpgpu_core_prepare_fp_op(lane, true, rm_field) < 0) {
                return -1;
            }
            gpgpu_core_set_fpr_s(lane, rd,
                                 uint32_to_float32(lane->gpr[rs1],
                                                   &lane->fp_status));
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: unsupported FCVT.S.* rs2=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          rs2, inst, pc);
            return -1;
        }
    case GPGPU_RV_FUNCT7_FMV_X_W:
        if (rs2 != GPGPU_RV_FP_RS2_FMV) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: illegal FMV.X.W/FCLASS.S rs2=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          rs2, inst, pc);
            return -1;
        }
        switch (funct3) {
        case GPGPU_RV_FP_FUNCT3_FMV_X_W:
            lane->gpr[rd] = lane->fpr[rs1];
            return 0;
        case GPGPU_RV_FP_FUNCT3_FCLASS_S:
            lane->gpr[rd] = gpgpu_core_fclass_s(frs1);
            return 0;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: unsupported FMV.X.W/FCLASS.S funct3=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          funct3, inst, pc);
            return -1;
        }
    case GPGPU_RV_FUNCT7_FMV_W_X:
        if (rs2 != GPGPU_RV_FP_RS2_FMV ||
            funct3 != GPGPU_RV_FP_FUNCT3_FMV_W_X) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: illegal FMV.W.X encoding funct3=%u rs2=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          funct3, rs2, inst, pc);
            return -1;
        }
        lane->fpr[rd] = lane->gpr[rs1];
        return 0;
    case GPGPU_RV_FUNCT7_FCVT_BF16:
        switch (rs2) {
        case GPGPU_RV_FP_RS2_S_BF16:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            gpgpu_core_set_fpr_s(lane, rd,
                                 bfloat16_to_float32(lane->fpr[rs1] &
                                                     GPGPU_CORE_BF16_MASK,
                                                     &lane->fp_status));
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        case GPGPU_RV_FP_RS2_BF16_S:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            lane->fpr[rd] = float32_to_bfloat16(frs1, &lane->fp_status);
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: unsupported BF16 conversion rs2=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          rs2, inst, pc);
            return -1;
        }
    case GPGPU_RV_FUNCT7_FCVT_FP8:
        switch (rs2) {
        case GPGPU_RV_FP_RS2_S_E4M3:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            gpgpu_core_set_fpr_s(lane, rd,
                                 bfloat16_to_float32(
                                     float8_e4m3_to_bfloat16(
                                         lane->fpr[rs1] & GPGPU_CORE_FP8_MASK,
                                         &lane->fp_status),
                                     &lane->fp_status));
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        case GPGPU_RV_FP_RS2_E4M3_S:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            lane->fpr[rd] = float32_to_float8_e4m3(frs1, true,
                                                   &lane->fp_status);
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        case GPGPU_RV_FP_RS2_S_E5M2:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            gpgpu_core_set_fpr_s(lane, rd,
                                 bfloat16_to_float32(
                                     float8_e5m2_to_bfloat16(
                                         lane->fpr[rs1] & GPGPU_CORE_FP8_MASK,
                                         &lane->fp_status),
                                     &lane->fp_status));
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        case GPGPU_RV_FP_RS2_E5M2_S:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            lane->fpr[rd] = float32_to_float8_e5m2(frs1, true,
                                                   &lane->fp_status);
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: unsupported FP8 conversion rs2=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          rs2, inst, pc);
            return -1;
        }
    case GPGPU_RV_FUNCT7_FCVT_E2M1:
        switch (rs2) {
        case GPGPU_RV_FP_RS2_S_E2M1:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            gpgpu_core_set_fpr_s(lane, rd,
                                 gpgpu_core_e2m1_to_float32(
                                     lane->fpr[rs1] & GPGPU_CORE_FP4_MASK,
                                     &lane->fp_status));
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        case GPGPU_RV_FP_RS2_E2M1_S:
            if (gpgpu_core_prepare_fp_op(lane, false, 0) < 0) {
                return -1;
            }
            lane->fpr[rd] = gpgpu_core_float32_to_e2m1(frs1);
            gpgpu_core_sync_fcsr_from_fp_status(lane);
            return 0;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: unsupported E2M1 conversion rs2=%u "
                          "inst=0x%08x pc=0x%08x\n",
                          rs2, inst, pc);
            return -1;
        }
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: unsupported OP-FP funct7=0x%x rs2=%u "
                      "inst=0x%08x pc=0x%08x\n",
                      funct7, rs2, inst, pc);
        return -1;
    }
}

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
    uint32_t warp_width = MIN(s->warp_size, (uint32_t)GPGPU_WARP_SIZE);
    uint32_t total_cycles = 0;
    uint32_t lane_idx;

    // Serial lane execution, not parallel execution.
    for (lane_idx = 0; lane_idx < warp_width; lane_idx++) {
        GPGPULane *lane = &warp->lanes[lane_idx];

        if (!lane->active || !(warp->active_mask & (1u << lane_idx))) {
            continue;
        }

        gpgpu_core_set_lane_context(s, warp, lane_idx);

        while (lane->active) {
            uint32_t inst;
            uint32_t pc = lane->pc;
            uint32_t next_pc = pc + 4;
            uint32_t opcode;
            uint32_t rd, rs1, rs2, funct3, funct7;
            uint32_t rs1_val, rs2_val;

            if (total_cycles++ >= max_cycles) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "gpgpu: warp exceeded max cycles (%u)\n",
                              max_cycles);
                return -1;
            }

            if (pc & GPGPU_RV_PC_ALIGN_MASK) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "gpgpu: misaligned instruction fetch pc=0x%08x\n",
                              pc);
                return -1;
            }

            if (pc >= GPGPU_CORE_CTRL_BASE) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "gpgpu: instruction fetch from ctrl space "
                              "pc=0x%08x\n", pc);
                return -1;
            }

            if (gpgpu_core_mem_read(s, pc, GPGPU_CORE_MEM_SIZE_WORD, &inst) < 0) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "gpgpu: instruction fetch failed pc=0x%08x\n", pc);
                return -1;
            }

            opcode = extract32(inst, 0, 7);
            rd = GET_RD(inst);
            rs1 = GET_RS1(inst);
            rs2 = GET_RS2(inst);
            funct3 = GET_FUNCT3(inst);
            funct7 = GET_FUNCT7(inst);
            rs1_val = lane->gpr[rs1];
            rs2_val = lane->gpr[rs2];

            switch (opcode) {
            case GPGPU_RV_OPCODE_LUI:
                lane->gpr[rd] = inst & GPGPU_RV_U_IMM_MASK;
                break;
            case GPGPU_RV_OPCODE_AUIPC:
                lane->gpr[rd] = pc + (inst & GPGPU_RV_U_IMM_MASK);
                break;
            case GPGPU_RV_OPCODE_JAL:
                lane->gpr[rd] = next_pc;
                next_pc = pc + (int32_t)GET_JAL_IMM(inst);
                break;
            case GPGPU_RV_OPCODE_JALR:
                if (funct3 != GPGPU_RV_FUNCT3_ADD_SUB) {
                    qemu_log_mask(LOG_GUEST_ERROR,
                                  "gpgpu: illegal jalr funct3=%u\n", funct3);
                    return -1;
                }
                lane->gpr[rd] = next_pc;
                next_pc = (rs1_val + sextract32(inst, 20, 12)) &
                          GPGPU_RV_JALR_TARGET_MASK;
                break;
            case GPGPU_RV_OPCODE_BRANCH: {
                GPGPURVBranchInsn branch = gpgpu_core_decode_branch_insn(funct3);
                bool take = false;
                int32_t lhs = (int32_t)rs1_val;
                int32_t rhs = (int32_t)rs2_val;

                switch (branch) {
                case GPGPU_RV_BRANCH_BEQ:
                    take = rs1_val == rs2_val;
                    break;
                case GPGPU_RV_BRANCH_BNE:
                    take = rs1_val != rs2_val;
                    break;
                case GPGPU_RV_BRANCH_BLT:
                    take = lhs < rhs;
                    break;
                case GPGPU_RV_BRANCH_BGE:
                    take = lhs >= rhs;
                    break;
                case GPGPU_RV_BRANCH_BLTU:
                    take = rs1_val < rs2_val;
                    break;
                case GPGPU_RV_BRANCH_BGEU:
                    take = rs1_val >= rs2_val;
                    break;
                default:
                    qemu_log_mask(LOG_GUEST_ERROR,
                                  "gpgpu: illegal branch funct3=%u\n", funct3);
                    return -1;
                }

                if (take) {
                    next_pc = pc + (int32_t)GET_B_IMM(inst);
                }
                break;
            }
            case GPGPU_RV_OPCODE_LOAD: {
                GPGPURVLoadInsn load = gpgpu_core_decode_load_insn(funct3);
                uint32_t mem_val;
                uint32_t addr = rs1_val + sextract32(inst, 20, 12);

                switch (load) {
                case GPGPU_RV_LOAD_LB:
                    if (gpgpu_core_mem_read(s, addr, GPGPU_CORE_MEM_SIZE_BYTE,
                                            &mem_val) < 0) {
                        return -1;
                    }
                    lane->gpr[rd] = (int8_t)mem_val;
                    break;
                case GPGPU_RV_LOAD_LH:
                    if (gpgpu_core_mem_read(s, addr, GPGPU_CORE_MEM_SIZE_HALF,
                                            &mem_val) < 0) {
                        return -1;
                    }
                    lane->gpr[rd] = (int16_t)mem_val;
                    break;
                case GPGPU_RV_LOAD_LW:
                    if (gpgpu_core_mem_read(s, addr, GPGPU_CORE_MEM_SIZE_WORD,
                                            &mem_val) < 0) {
                        return -1;
                    }
                    lane->gpr[rd] = mem_val;
                    break;
                case GPGPU_RV_LOAD_LBU:
                    if (gpgpu_core_mem_read(s, addr, GPGPU_CORE_MEM_SIZE_BYTE,
                                            &mem_val) < 0) {
                        return -1;
                    }
                    lane->gpr[rd] = mem_val & 0xff;
                    break;
                case GPGPU_RV_LOAD_LHU:
                    if (gpgpu_core_mem_read(s, addr, GPGPU_CORE_MEM_SIZE_HALF,
                                            &mem_val) < 0) {
                        return -1;
                    }
                    lane->gpr[rd] = mem_val & 0xffff;
                    break;
                default:
                    qemu_log_mask(LOG_GUEST_ERROR,
                                  "gpgpu: illegal load funct3=%u\n", funct3);
                    return -1;
                }
                break;
            }
            case GPGPU_RV_OPCODE_STORE: {
                GPGPURVStoreInsn store = gpgpu_core_decode_store_insn(funct3);
                uint32_t addr = rs1_val + (int32_t)GET_STORE_IMM(inst);
                unsigned size;

                switch (store) {
                case GPGPU_RV_STORE_SB:
                    size = GPGPU_CORE_MEM_SIZE_BYTE;
                    break;
                case GPGPU_RV_STORE_SH:
                    size = GPGPU_CORE_MEM_SIZE_HALF;
                    break;
                case GPGPU_RV_STORE_SW:
                    size = GPGPU_CORE_MEM_SIZE_WORD;
                    break;
                default:
                    qemu_log_mask(LOG_GUEST_ERROR,
                                  "gpgpu: illegal store funct3=%u\n", funct3);
                    return -1;
                }

                if (gpgpu_core_mem_write(s, addr, rs2_val, size) < 0) {
                    return -1;
                }
                break;
            }
            case GPGPU_RV_OPCODE_OP_IMM: {
                GPGPURVOpImmInsn op_imm =
                    gpgpu_core_decode_op_imm_insn(funct3, funct7);
                int32_t imm = sextract32(inst, 20, 12);

                switch (op_imm) {
                case GPGPU_RV_OP_IMM_ADDI:
                    lane->gpr[rd] = rs1_val + imm;
                    break;
                case GPGPU_RV_OP_IMM_SLLI:
                    lane->gpr[rd] = rs1_val << extract32(inst, 20, 5);
                    break;
                case GPGPU_RV_OP_IMM_SLTI:
                    lane->gpr[rd] = (int32_t)rs1_val < imm;
                    break;
                case GPGPU_RV_OP_IMM_SLTIU:
                    lane->gpr[rd] = rs1_val < (uint32_t)imm;
                    break;
                case GPGPU_RV_OP_IMM_XORI:
                    lane->gpr[rd] = rs1_val ^ imm;
                    break;
                case GPGPU_RV_OP_IMM_SRLI:
                    lane->gpr[rd] = rs1_val >> extract32(inst, 20, 5);
                    break;
                case GPGPU_RV_OP_IMM_SRAI:
                    lane->gpr[rd] = (int32_t)rs1_val >>
                                    extract32(inst, 20, 5);
                    break;
                case GPGPU_RV_OP_IMM_ORI:
                    lane->gpr[rd] = rs1_val | imm;
                    break;
                case GPGPU_RV_OP_IMM_ANDI:
                    lane->gpr[rd] = rs1_val & imm;
                    break;
                default:
                    qemu_log_mask(LOG_GUEST_ERROR,
                                  "gpgpu: illegal op-imm funct3=%u\n", funct3);
                    return -1;
                }
                break;
            }
            case GPGPU_RV_OPCODE_OP:
            {
                GPGPURVOpInsn op = gpgpu_core_decode_op_insn(funct3, funct7);

                switch (op) {
                case GPGPU_RV_OP_ADD:
                    lane->gpr[rd] = rs1_val + rs2_val;
                    break;
                case GPGPU_RV_OP_SUB:
                    lane->gpr[rd] = rs1_val - rs2_val;
                    break;
                case GPGPU_RV_OP_SLL:
                    lane->gpr[rd] = rs1_val << (rs2_val & GPGPU_RV_SHAMT_MASK);
                    break;
                case GPGPU_RV_OP_SLT:
                    lane->gpr[rd] = (int32_t)rs1_val < (int32_t)rs2_val;
                    break;
                case GPGPU_RV_OP_SLTU:
                    lane->gpr[rd] = rs1_val < rs2_val;
                    break;
                case GPGPU_RV_OP_XOR:
                    lane->gpr[rd] = rs1_val ^ rs2_val;
                    break;
                case GPGPU_RV_OP_SRL:
                    lane->gpr[rd] = rs1_val >> (rs2_val & GPGPU_RV_SHAMT_MASK);
                    break;
                case GPGPU_RV_OP_SRA:
                    lane->gpr[rd] = (int32_t)rs1_val >>
                                    (rs2_val & GPGPU_RV_SHAMT_MASK);
                    break;
                case GPGPU_RV_OP_OR:
                    lane->gpr[rd] = rs1_val | rs2_val;
                    break;
                case GPGPU_RV_OP_AND:
                    lane->gpr[rd] = rs1_val & rs2_val;
                    break;
                case GPGPU_RV_OP_MUL:
                    lane->gpr[rd] = (uint32_t)((uint64_t)rs1_val * rs2_val);
                    break;
                default:
                    qemu_log_mask(LOG_GUEST_ERROR,
                                  "gpgpu: illegal op funct3=%u funct7=0x%x\n",
                                  funct3, funct7);
                    return -1;
                }
                break;
            }
            case GPGPU_RV_OPCODE_LOAD_FP:
            case GPGPU_RV_OPCODE_STORE_FP:
                if (gpgpu_core_exec_fp_load_store(s, lane, inst, pc, opcode,
                                                  funct3, rd, rs1_val, rs2) < 0) {
                    return -1;
                }
                break;
            case GPGPU_RV_OPCODE_FMADD:
            case GPGPU_RV_OPCODE_FMSUB:
            case GPGPU_RV_OPCODE_FNMSUB:
            case GPGPU_RV_OPCODE_FNMADD:
                if (gpgpu_core_exec_fp_fma(lane, inst, pc, opcode, rd, rs1,
                                           rs2) < 0) {
                    return -1;
                }
                break;
            case GPGPU_RV_OPCODE_FP:
                if (gpgpu_core_exec_op_fp(lane, inst, pc, rd, rs1, rs2,
                                          funct3, funct7) < 0) {
                    return -1;
                }
                break;
            case GPGPU_RV_OPCODE_SYSTEM:
            {
                GPGPURVSystemInsn sys = gpgpu_core_decode_system_insn(inst,
                                                                      funct3);

                if (sys == GPGPU_RV_SYSTEM_EBREAK) {
                    lane->pc = next_pc;
                    lane->active = false;
                    break;
                }

                switch (sys) {
                case GPGPU_RV_SYSTEM_CSRRW:
                case GPGPU_RV_SYSTEM_CSRRS:
                case GPGPU_RV_SYSTEM_CSRRC:
                case GPGPU_RV_SYSTEM_CSRRWI:
                case GPGPU_RV_SYSTEM_CSRRSI:
                case GPGPU_RV_SYSTEM_CSRRCI: {
                    uint32_t csr = extract32(inst, 20, 12);
                    uint32_t old_csr;
                    uint32_t write_val = 0;
                    bool do_write = false;

                    if (gpgpu_core_read_csr(lane, csr, &old_csr) < 0) {
                        return -1;
                    }

                    switch (sys) {
                    case GPGPU_RV_SYSTEM_CSRRW:
                        write_val = rs1_val;
                        do_write = true;
                        break;
                    case GPGPU_RV_SYSTEM_CSRRS:
                        write_val = old_csr | rs1_val;
                        do_write = rs1 != 0;
                        break;
                    case GPGPU_RV_SYSTEM_CSRRC:
                        write_val = old_csr & ~rs1_val;
                        do_write = rs1 != 0;
                        break;
                    case GPGPU_RV_SYSTEM_CSRRWI:
                        write_val = rs1;
                        do_write = true;
                        break;
                    case GPGPU_RV_SYSTEM_CSRRSI:
                        write_val = old_csr | rs1;
                        do_write = rs1 != 0;
                        break;
                    case GPGPU_RV_SYSTEM_CSRRCI:
                        write_val = old_csr & ~rs1;
                        do_write = rs1 != 0;
                        break;
                    default:
                        g_assert_not_reached();
                    }

                    lane->gpr[rd] = old_csr;
                    if (do_write && gpgpu_core_write_csr(lane, csr, write_val) < 0) {
                        return -1;
                    }
                    break;
                }
                default:
                    qemu_log_mask(LOG_GUEST_ERROR,
                                  "gpgpu: unsupported system decode funct3=%u "
                                  "inst=0x%08x "
                                  "pc=0x%08x\n", funct3, inst, pc);
                    return -1;
                }
                break;
            }
            default:
                qemu_log_mask(LOG_GUEST_ERROR,
                              "gpgpu: unsupported opcode=0x%02x inst=0x%08x "
                              "pc=0x%08x\n",
                              opcode, inst, pc);
                return -1;
            }

            lane->gpr[0] = 0;
            if (lane->active) {
                lane->pc = next_pc;
            }
        }
    }

    return 0;
}

int gpgpu_core_exec_kernel(GPGPUState *s)
{
    // soc mannul set
    static const uint32_t max_cycles_per_warp = 800000;
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

                    gpgpu_core_set_lane_context(s, &warp, 0);

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
