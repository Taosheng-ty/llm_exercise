"""
Exercise 01: GPU Placement for Actor-Critic Training - Solution
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
    """
    effective_critic = critic_gpu_count if use_critic else 0

    # Calculate total GPUs needed
    if colocate:
        # Rollout shares actor GPUs, so rollout doesn't need extra GPUs.
        # But actor must have at least rollout_gpu_count GPUs.
        if actor_gpu_count < rollout_gpu_count:
            raise ValueError(
                f"Colocate mode requires actor_gpu_count ({actor_gpu_count}) "
                f">= rollout_gpu_count ({rollout_gpu_count})"
            )
        needed = actor_gpu_count + effective_critic
    else:
        needed = actor_gpu_count + effective_critic + rollout_gpu_count

    if needed > total_gpus:
        raise ValueError(
            f"Not enough GPUs: need {needed} but only {total_gpus} available"
        )

    # Allocate in order: actor, critic, rollout
    offset = 0
    actor_gpus = list(range(offset, offset + actor_gpu_count))
    offset += actor_gpu_count

    if use_critic:
        critic_gpus = list(range(offset, offset + effective_critic))
        offset += effective_critic
    else:
        critic_gpus = None

    if colocate:
        # Rollout shares actor GPUs (first rollout_gpu_count of actor)
        rollout_gpus = list(actor_gpus[:rollout_gpu_count])
    else:
        rollout_gpus = list(range(offset, offset + rollout_gpu_count))

    return {
        "actor": actor_gpus,
        "critic": critic_gpus,
        "rollout": rollout_gpus,
    }
