"""
Exercise 01: GPU Placement for Actor-Critic Training

In distributed RL training (e.g., RLHF with PPO), we need to allocate GPUs to
different roles: actor (policy model), critic (value model), and rollout (inference
engine for generating trajectories).

Inspired by slime's placement_group.py create_placement_groups(), implement a
function that allocates non-overlapping GPU sets to each role, with an option
to colocate the rollout engine on the same GPUs as the actor.

Rules:
- GPUs are identified by integer IDs from 0 to total_gpus - 1.
- actor_gpu_count, critic_gpu_count, rollout_gpu_count are the number of GPUs
  each role needs.
- If colocate=True, rollout shares the actor's GPUs (no extra GPUs needed for
  rollout), so total needed = actor + critic.
- If colocate=False, each role gets its own non-overlapping set, so total
  needed = actor + critic + rollout.
- GPUs are assigned in order: actor gets [0, actor_count), critic gets
  [actor_count, actor_count + critic_count), rollout gets the remainder
  (or actor's GPUs if colocated).
- If use_critic=False, no GPUs are allocated for the critic (critic_gpu_count
  is ignored and treated as 0).
- Raise ValueError if not enough GPUs are available.

Returns a dict:
  {
    "actor": [list of GPU IDs],
    "critic": [list of GPU IDs] or None,
    "rollout": [list of GPU IDs],
  }
"""


def allocate_gpus(
    total_gpus: int,
    actor_gpu_count: int,
    critic_gpu_count: int,
    rollout_gpu_count: int,
    colocate: bool = False,
    use_critic: bool = True,
) -> dict:
    """Allocate GPU IDs to actor, critic, and rollout roles.

    Args:
        total_gpus: Total number of available GPUs (IDs 0..total_gpus-1).
        actor_gpu_count: Number of GPUs the actor needs.
        critic_gpu_count: Number of GPUs the critic needs (ignored if use_critic=False).
        rollout_gpu_count: Number of GPUs the rollout engine needs.
        colocate: If True, rollout shares the actor's GPUs.
        use_critic: If False, skip critic allocation entirely.

    Returns:
        Dict mapping role name -> list of GPU IDs (or None for critic if unused).

    Raises:
        ValueError: If total_gpus is insufficient for the requested allocation.
            Message format: "Not enough GPUs: need {needed} but only {total_gpus} available"
        ValueError: If colocate=True and actor_gpu_count < rollout_gpu_count.
            Message format: "Colocate mode requires actor_gpu_count ({actor_gpu_count}) >= rollout_gpu_count ({rollout_gpu_count})"
    """
    # TODO: Implement GPU allocation logic
    raise NotImplementedError("Implement allocate_gpus")
