"""
Realization lifecycle: the stages one realization passes through across invocations
(fresh start, resume, extend, overwrite, complete), for any task kind (ADR 0003).

    outcome = realization_lifecycle.run(record, trainer, hparams=..., critical_keys=...,
                                        code_files=..., target_steps=N, overwrite=False,
                                        stop_rule=SubtaskLossRatioBelow(1e-5),
                                        on_mismatch=refuse)

run() owns everything around training: the critical-hyperparameter check against what the
realization recorded, the overwrite wipe, training meta / provenance / code snapshot,
resume-or-fresh, how many steps remain, and the final save. The trainer is an adapter that
knows only its physics:

    trainer.prepare(record)        build or load the network and task, writing them through
                                   the record
    trainer.fresh_state()          TrainState of a realization that has not trained yet
    trainer.train(state, n_steps, stop_rule, on_checkpoint) -> TrainResult
                                   on_checkpoint is record.save_checkpoint
"""
from dataclasses import dataclass

import numpy as np

from base.realization_record import BestSoFar, Loss  # noqa: F401  (part of the trainer contract)
from training.src.run_provenance import (
    has_critical_mismatch, save_training_meta, save_run_provenance, save_code_snapshot,
)

STAGES = ('fresh', 'resume', 'extend', 'overwrite', 'complete')
STATUSES = ('converged', 'steps_exhausted', 'nan')


@dataclass(frozen=True)
class TrainState:
    """Where training stands: steps completed, the stiffnesses the next step starts from,
    the loss history, and the trainer's best-so-far state (None before the first step)."""
    step: int
    stiffnesses: np.ndarray
    loss: Loss
    best: object = None   # BestSoFar or None


@dataclass(frozen=True)
class TrainResult:
    state: TrainState
    status: str   # one of STATUSES


@dataclass(frozen=True)
class Outcome:
    """What one invocation did. stage: how it started (STAGES), or 'refused' after a
    critical-hyperparameter mismatch without overwrite. status: how training ended
    (STATUSES), or None when no training step ran. state: the state saved last."""
    stage: str
    status: str = None
    state: TrainState = None
    refusal: object = None   # what on_mismatch returned, when stage == 'refused'


@dataclass(frozen=True)
class SubtaskLossRatioBelow:
    """Stop rule: every subtask's loss has fallen below `threshold` times its first value."""
    threshold: float

    def reached(self, first, now):
        """first, now: per-subtask losses at step 0 and at the current step."""
        return bool(np.all(np.asarray(now) / np.asarray(first) < self.threshold))


def run(record, trainer, *, hparams, critical_keys, code_files, target_steps, overwrite,
        stop_rule, on_mismatch):
    """Take `record`'s realization from wherever it stands to `target_steps` steps (or the
    stop rule). on_mismatch(record) is called when the recorded critical hyperparameters
    differ from `hparams` and overwrite is False: it may raise, else its return value ends
    the invocation as Outcome('refused')."""
    run_dir = record.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    stage = None
    if has_critical_mismatch(run_dir, hparams, critical_keys=critical_keys):
        if not overwrite:
            return Outcome('refused', refusal=on_mismatch(record))
        print(f"  --overwrite set: recorded hyperparameters differ from this run's — wiping "
              f"{run_dir} and restarting from scratch under the new hyperparameters.")
        record.wipe()
        stage = 'overwrite'
    previous_target = (record.config.get('n_training_steps') if record.has('config') else None)

    save_training_meta(run_dir, hparams, critical_keys=critical_keys)
    save_run_provenance(run_dir, extra=hparams)
    save_code_snapshot(run_dir, code_files)
    record.refresh()

    trainer.prepare(record)

    resume = record.resume_state()
    best = record.best_so_far if record.has('best_so_far') else None
    if resume is None:
        state = trainer.fresh_state()
        state = TrainState(state.step, state.stiffnesses, state.loss, best)
        stage = stage or 'fresh'
    else:
        state = TrainState(resume.step, resume.stiffnesses, resume.loss, best)
    if best is not None:
        print(f"  Best state    : combined_mse={best.training_loss:.4e} (loaded from disk)")

    remaining = max(0, target_steps - state.step)
    if stage is None:
        if remaining == 0:
            stage = 'complete'
        elif previous_target is not None and state.step < previous_target:
            stage = 'resume'
        else:
            stage = 'extend'

    status = None
    if remaining > 0:
        result = trainer.train(state, remaining, stop_rule, record.save_checkpoint)
        state, status = result.state, result.status

    record.save_final(state.stiffnesses, state.loss, state.best)
    return Outcome(stage, status, state)
