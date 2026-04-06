"""
Exercise 03: Async Training Scheduler

In slime's train_async.py, the training loop overlaps rollout generation with
model training: while the current batch trains, the next rollout is already
being generated. This pattern is key to efficient distributed RL training.

Implement a task scheduler that models this overlap. Each task has:
  - name: unique string identifier
  - task_type: "generate" or "train"
  - duration: how many time units the task takes
  - dependencies: list of task names that must complete before this task starts

The scheduler should:
1. Start tasks as soon as all their dependencies are complete.
2. Allow at most one "generate" task and one "train" task to run concurrently
   (they use different resources: generate uses rollout GPUs, train uses
   training GPUs). But two "generate" tasks cannot overlap, and two "train"
   tasks cannot overlap.
3. Return the schedule and total makespan (time of last task completion).

This models the real async pattern: generate(i+1) overlaps with train(i).
"""

from dataclasses import dataclass


@dataclass
class Task:
    name: str
    task_type: str  # "generate" or "train"
    duration: int
    dependencies: list[str]


def schedule_tasks(tasks: list[Task]) -> tuple[list[tuple[str, int, int]], int]:
    """Schedule tasks to minimize total time with resource constraints.

    Rules:
    - A task can start only after ALL its dependencies have finished.
    - At most one "generate" task can run at a time.
    - At most one "train" task can run at a time.
    - A "generate" and a "train" task CAN run simultaneously.

    Args:
        tasks: List of Task objects to schedule.

    The scheduler should produce an optimal schedule (minimum makespan).

    Tie-breaking: when multiple tasks are eligible to start at the same time
    on the same resource, choose alphabetically by task name.

    Returns:
        A tuple of:
          - Schedule: list of (task_name, start_time, end_time) sorted by start_time,
            then by task_name for ties.
          - Total makespan: the end time of the last task to finish.
    """
    # TODO: Implement the async training scheduler
    raise NotImplementedError("Implement schedule_tasks")
