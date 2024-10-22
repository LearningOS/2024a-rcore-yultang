//! Types related to task management

use crate::config::MAX_SYSCALL_NUM;

use super::TaskContext;

/// The task control block (TCB) of a task.
#[derive(Copy, Clone)]
pub struct TaskControlBlock {
    /// The task status in it's lifecycle
    pub task_status: TaskStatus,

    /// The task context
    pub task_cx: TaskContext,

    /// The numbers of every sys_call occurs
    pub syscall_times: [u32; MAX_SYSCALL_NUM],

    /// Timing of this task starts to be scheduled
    pub start_time: usize,

    /// Mark this task if begins
    pub begin: bool,
}

/// The status of a task
#[derive(Copy, Clone, PartialEq)]
pub enum TaskStatus {
    /// uninitialized
    UnInit,
    /// ready to run
    Ready,
    /// running
    Running,
    /// exited
    Exited,
}
