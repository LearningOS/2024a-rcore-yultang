//! Loading user applications into memory
//!
//! For chapter 3, user applications are simply part of the data included in the
//! kernel binary, so we only need to copy them to the space allocated for each
//! app to load them. We also allocate fixed spaces for each task's
//! [`KernelStack`] and [`UserStack`].

use crate::config::*;
use crate::trap::TrapContext;
use core::arch::asm;

#[repr(align(4096))]
#[derive(Copy, Clone)]
struct KernelStack {
    data: [u8; KERNEL_STACK_SIZE],
}

#[repr(align(4096))]
#[derive(Copy, Clone)]
struct UserStack {
    data: [u8; USER_STACK_SIZE],
}

static KERNEL_STACK: [KernelStack; MAX_APP_NUM] = [KernelStack {
    data: [0; KERNEL_STACK_SIZE],
}; MAX_APP_NUM];

static USER_STACK: [UserStack; MAX_APP_NUM] = [UserStack {
    data: [0; USER_STACK_SIZE],
}; MAX_APP_NUM];

impl KernelStack {
    fn get_sp(&self) -> usize {
        self.data.as_ptr() as usize + KERNEL_STACK_SIZE
    }
    pub fn push_context(&self, trap_cx: TrapContext) -> usize {
        let trap_cx_ptr = (self.get_sp() - core::mem::size_of::<TrapContext>()) as *mut TrapContext;
        unsafe {
            *trap_cx_ptr = trap_cx;
        }
        trap_cx_ptr as usize
    }
}

impl UserStack {
    fn get_sp(&self) -> usize {
        self.data.as_ptr() as usize + USER_STACK_SIZE
    }
}

/// Get base address of app i.
fn get_base_i(app_id: usize) -> usize {
    APP_BASE_ADDRESS + app_id * APP_SIZE_LIMIT
}

/// Get the total number of applications.
pub fn get_num_app() -> usize {
    extern "C" {
        fn _num_app();
    }
    unsafe { (_num_app as usize as *const usize).read_volatile() }
}

/// Load nth user app at
/// [APP_BASE_ADDRESS + n * APP_SIZE_LIMIT, APP_BASE_ADDRESS + (n+1) * APP_SIZE_LIMIT].
pub fn load_apps() {
    extern "C" {
        fn _num_app();
    }
    
    // num_app_ptr为裸指针, 指向_num_app的地址
    let num_app_ptr = _num_app as usize as *const usize;
    let num_app = get_num_app();
    let app_start = unsafe { 
        // num_app_ptr.add(1)表示向前移动一个元素大小(usize)
        // num_app + 1表示切片长度
        // num_app_ptr是_num_app函数的地址, 不包含有用的数据
        // 实际的数据从num_app_ptr.add(1)开始
        core::slice::from_raw_parts(num_app_ptr.add(1), num_app + 1) 
    };

    // clear i-cache first
    unsafe {
        asm!("fence.i");
    }
    // load apps
    for i in 0..num_app {
        let base_i = get_base_i(i);

        // 1. 迭代器生成: (base_i..base_i + APP_SIZE_LIMIT): 生成一个从base_i到base_i + APP_SIZE_LIMIT - 1的整数序列 
        // 2. .for_each(|addr| ...): 迭代器的一个方法, 即对迭代器中的每个元素执行指定闭包操作
        // 3. unsafe: 用来声明代码中包含不安全代码块
        // 4. .write_volatile(0): 将0写入由裸指针addr as *mut u8指向的内存地址, 并告诉编译器不要优化这个写操作(volatile)
        // region clear to all zero
        (base_i..base_i + APP_SIZE_LIMIT)
            .for_each(|addr| unsafe { (addr as *mut u8).write_volatile(0) });

        // load app from data section to memory
        let src = unsafe {
            // app_start[i]是第i个app的起始地址
            // app_statr[i+1]是第i+1个app的起始地址
            // 所以两者相减就是第i个app的大小
            core::slice::from_raw_parts(app_start[i] as *const u8, app_start[i + 1] - app_start[i])
        };

        let dst = unsafe { 
            core::slice::from_raw_parts_mut(base_i as *mut u8, src.len()) 
        };

        // 标准库函数
        dst.copy_from_slice(src);
    }
}

/// get app info with entry and sp and save `TrapContext` in kernel stack
pub fn init_app_cx(app_id: usize) -> usize {
    KERNEL_STACK[app_id].push_context(TrapContext::app_init_context(
        get_base_i(app_id),
        USER_STACK[app_id].get_sp(),
    ))
}
