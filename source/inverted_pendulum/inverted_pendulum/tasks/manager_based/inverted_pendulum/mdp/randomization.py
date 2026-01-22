# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Callable

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase


def _resolve_randomizer(
    env: ManagerBasedEnv, randomize_func: Callable, randomize_kwargs: dict
) -> Callable[[torch.Tensor | None], dict | None]:
    """Return a callable that can be invoked for randomization.

    The helper supports both function-based and class-based (``ManagerTermBase``) randomizers.
    For class-based terms, we lazily instantiate once and cache per environment to avoid
    repeatedly constructing heavy terms on every reset.
    """

    # class-based term
    if isinstance(randomize_func, type) and issubclass(randomize_func, ManagerTermBase):
        cache = getattr(env, "_randomize_term_cache", None)
        if cache is None:
            cache = {}
            setattr(env, "_randomize_term_cache", cache)

        key = randomize_func.__name__
        if key not in cache:
            term_cfg = EventTermCfg(func=randomize_func, params=randomize_kwargs)
            cache[key] = randomize_func(cfg=term_cfg, env=env)

        term_impl = cache[key]

        def _call(env_ids: torch.Tensor | None):
            return term_impl(env, env_ids, **randomize_kwargs)

        return _call

    # function-based term
    def _call(env_ids: torch.Tensor | None):
        return randomize_func(env, env_ids, **randomize_kwargs)

    return _call


def randomize_if_survived(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    survival_fraction: float,
    randomize_func: Callable,
    randomize_kwargs: dict,
):
    """Gate randomization until environments have survived a fraction of the episode.

    Args:
        env: The environment instance.
        env_ids: Environment ids to consider. If None, all envs.
        survival_fraction: Fraction (0-1] of max episode length to consider "survived".
        randomize_func: Underlying randomization function to call.
        randomize_kwargs: Keyword arguments forwarded to `randomize_func`.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # compute episode length in steps
    max_len = getattr(env, "max_episode_length", None)
    if max_len is None:
        # best-effort fallback from cfg
        dt = getattr(env.sim, "dt", None)
        decimation = getattr(env.cfg, "decimation", 1)
        episode_len_s = getattr(env.cfg, "episode_length_s", 0.0)
        if dt is not None and episode_len_s > 0.0:
            max_len = int(episode_len_s / dt / decimation)
        else:
            max_len = 0

    if max_len <= 0:
        # no gating if we cannot infer length
        gated_env_ids = env_ids
    else:
        progress_buf = getattr(env, "progress_buf", None)
        if progress_buf is None:
            progress_buf = getattr(env, "episode_length_buf", None)
        if progress_buf is None:
            return {}
        progress = progress_buf[env_ids]
        mask = progress >= survival_fraction * max_len
        if not torch.any(mask):
            return {}
        gated_env_ids = env_ids[mask]

    randomizer = _resolve_randomizer(env, randomize_func, randomize_kwargs)
    return randomizer(gated_env_ids)


def randomize_if_success_rate(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    success_threshold: float,
    randomize_func: Callable,
    randomize_kwargs: dict,
    fallback_survival_fraction: float | None = None,
):
    """Gate randomization by a success rate signal, with optional survival fallback.

    The helper looks for success rate signals in ``env.extras`` under common keys and applies
    randomization only when the rate meets the threshold. If no rate is found and a fallback
    survival fraction is provided, it defers to :func:`randomize_if_survived`.
    """
    # pick global success rate if exposed
    success_rate = None
    extras = getattr(env, "extras", None)
    if isinstance(extras, dict):
        for key in ("success_rate", "success_fraction", "success"):
            if key in extras:
                success_rate = extras[key]
                break

        # Derive success if only termination stats are present:
        # success = timeout AND no illegal_contact. We approximate by
        #   success_rate = time_out_fraction
        if success_rate is None:
            term_dict = extras
            if "log" in extras and isinstance(extras["log"], dict):
                term_dict = extras["log"]
            t_out = term_dict.get("Episode_Termination/time_out", None)
            if t_out is not None:
                t_val = t_out.mean().item() if isinstance(t_out, torch.Tensor) else float(t_out)
                success_rate = t_val

    if success_rate is not None:
        # handle tensor or scalar
        if isinstance(success_rate, torch.Tensor):
            rate_val = success_rate.mean().item()
        else:
            rate_val = float(success_rate)
        if rate_val < success_threshold:
            return {}
        randomizer = _resolve_randomizer(env, randomize_func, randomize_kwargs)
        return randomizer(env_ids)

    if fallback_survival_fraction is not None:
        return randomize_if_survived(
            env=env,
            env_ids=env_ids,
            survival_fraction=fallback_survival_fraction,
            randomize_func=randomize_func,
            randomize_kwargs=randomize_kwargs,
        )

    # no signal -> skip
    return {}


def increment_terrain_level_if_success(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    success_threshold: float,
    increment: int = 1,
    max_level: int | None = None,
    log_key: str | None = "terrain_level",
):
    """Increment ``terrain.max_init_terrain_level`` when success rate is high enough.

    This mirrors the gating used for domain randomization so it stays consistent with
    the success signal used elsewhere (timeout with no illegal contact). The change
    is applied globally because terrain level affects all env origins, not per-env.
    """

    # derive success rate from extras using the same heuristics as randomization
    success_rate = None
    extras = getattr(env, "extras", None)
    if isinstance(extras, dict):
        for key in ("success_rate", "success_fraction", "success"):
            if key in extras:
                success_rate = extras[key]
                break

        if success_rate is None:
            term_dict = extras
            if "log" in extras and isinstance(extras["log"], dict):
                term_dict = extras["log"]
            t_out = term_dict.get("Episode_Termination/time_out", None)
            if t_out is not None:
                t_val = t_out.mean().item() if isinstance(t_out, torch.Tensor) else float(t_out)
                success_rate = t_val

    if success_rate is None:
        return {}

    rate_val = success_rate.mean().item() if isinstance(success_rate, torch.Tensor) else float(success_rate)
    if rate_val < success_threshold:
        return {}

    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return {}

    # resolve current level from multiple possible attributes; default to 0 if missing
    cfg = getattr(terrain, "cfg", None)
    cur_level = int(getattr(cfg, "max_init_terrain_level", 0))

    # infer maximum from generator rows if not provided
    max_allowed = max_level
    if max_allowed is None:
        tg = getattr(terrain, "terrain_generator", None)
        if tg is None and cfg is not None:
            tg = getattr(cfg, "terrain_generator", None)
        if tg is not None and hasattr(tg, "num_rows"):
            max_allowed = max(int(tg.num_rows) - 1, 0)

    if max_allowed is None:
        return {}

    new_level = min(max_allowed, cur_level + max(int(increment), 1))
    if new_level <= cur_level:
        return {}

    # propagate new level to common attributes
    if cfg is not None:
        for attr in ("max_init_terrain_level", "max_terrain_level"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, new_level)
    for attr in ("max_init_terrain_level", "max_terrain_level"):
        if hasattr(terrain, attr):
            setattr(terrain, attr, new_level)

    # if terrain levels/types and origins exist, resample levels up to the new cap and refresh env origins
    terrain_levels = getattr(terrain, "terrain_levels", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    origins = getattr(terrain, "terrain_origins", None)
    if terrain_levels is not None and terrain_types is not None and origins is not None:
        num_envs = terrain_levels.shape[0]
        device = terrain_levels.device
        new_levels = torch.randint(0, new_level + 1, (num_envs,), device=device)
        terrain.terrain_levels = new_levels
        if hasattr(terrain, "env_origins") and terrain.env_origins is not None:
            terrain.env_origins[:] = origins[new_levels, terrain_types]
            if hasattr(env.scene, "env_origins"):
                env.scene.env_origins[:] = terrain.env_origins

    # optionally log the level so it shows up in summaries
    if log_key and isinstance(extras, dict):
        extras[log_key] = new_level

    return {"max_init_terrain_level": new_level}
