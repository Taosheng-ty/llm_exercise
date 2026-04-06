"""
Exercise 03: Async Training Scheduler - Solution
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

    Returns:
        A tuple of:
          - Schedule: list of (task_name, start_time, end_time) sorted by start_time,
            then by task_name for ties.
          - Total makespan: the end time of the last task to finish.
    """
    task_map = {t.name: t for t in tasks}
    end_times = {}  # task_name -> end_time

    # Track when each resource (generate, train) becomes free
    resource_free_at = {"generate": 0, "train": 0}

    schedule = []

    # Process tasks in topological order, using a greedy approach.
    # We simulate time: at each step, pick the earliest-startable ready task.
    scheduled_set = set()
    remaining = set(t.name for t in tasks)

    while remaining:
        # Find all tasks whose dependencies are satisfied
        ready = []
        for name in remaining:
            task = task_map[name]
            if all(dep in scheduled_set for dep in task.dependencies):
                # Earliest this task can start
                dep_done = max(
                    (end_times[dep] for dep in task.dependencies), default=0
                )
                earliest = max(dep_done, resource_free_at[task.task_type])
                ready.append((earliest, name))

        if not ready:
            raise ValueError("Circular dependency or missing task detected")

        # Pick the task that can start earliest (break ties by name for determinism)
        ready.sort(key=lambda x: (x[0], x[1]))
        earliest_start, chosen_name = ready[0]

        task = task_map[chosen_name]
        start = earliest_start
        end = start + task.duration

        schedule.append((chosen_name, start, end))
        end_times[chosen_name] = end
        resource_free_at[task.task_type] = end
        scheduled_set.add(chosen_name)
        remaining.remove(chosen_name)

    # Sort by start_time, then name
    schedule.sort(key=lambda x: (x[1], x[0]))
    makespan = max(entry[2] for entry in schedule) if schedule else 0

    return schedule, makespan
