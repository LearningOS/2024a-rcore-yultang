.altmacro
.macro SAVE_SN n
    sd s\n, (\n+2)*8(a0)
.endm
.macro LOAD_SN n
    ld s\n, (\n+2)*8(a1)
.endm
    # 表示接下来代码将被放入代码段中
    .section .text

    # 定义全局函数 __switch
    .globl __switch

__switch:
    # __switch(
    #     current_task_cx_ptr: *mut TaskContext,
    #     next_task_cx_ptr: *const TaskContext
    # )
    
    # a0中存储的是current_task_cx_ptr
    # 将当前任务的 堆栈指针保存到当前任务上下文的结构体中
    # save kernel stack of current task
    sd sp, 8(a0)

    # 将当前任务的 返回地址寄存器ra(存储返回地址)保存到0(a0)
    sd ra, 0(a0)

    # 循环12次, 保存s0~s11
    .set n, 0
    .rept 12
        SAVE_SN %n
        .set n, n + 1
    .endr

    # 恢复另一个任务的上下文执行
    # restore ra & s0~s11 of next execution
    ld ra, 0(a1)

    .set n, 0
    .rept 12
        LOAD_SN %n
        .set n, n + 1
    .endr

    # restore kernel stack of next task
    ld sp, 8(a1)

    ret

