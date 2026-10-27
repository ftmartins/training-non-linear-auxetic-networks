"""
Realization lifecycle: the stages one realization passes through across invocations
(fresh start, resume, extend, overwrite, complete), for any task kind (ADR 0003).

    outcome = realization_lifecycle.run(record, trainer, hparams=..., critical_keys=...,
                                        code_files=..., target_steps=N, overwrite=False,
                                        stop_rule=SubtaskLossRatioBelow(1e-5),
                                        on_mismatch=refuse)

run() owns everything around training: the critical-hyperparameter check against what the
realization recorded, the overwrite wipe, the done check, resume-or-fresh, training meta /
provenance / code snapshot, how many steps remain, and when the final save happens. The
trainer is an adapter that knows only its physics:

    trainer.prepare(record)        build or load the network and task, writing them through
                                   the record
    trainer.fresh_state()          TrainState of a realization that has not trained yet
    trainer.train(state, n_steps, stop_rule, on_checkpoint) -> TrainResult
                                   on_checkpoint is record.save_checkpoint
  optional:
    trainer.resumed_state(resume)  TrainState from the record's ResumeState (default: its
                                   step, stiffnesses and loss, plus the best-so-far state)
    trainer.extra_hparams(state)   hyperparameters known only once the network is built,
                                   recorded with the others (auxetic network fingerprint)
    trainer.save_final(record, state, status)
                                   the invocation's final save (default:
                                   record.save_final(stiffnesses, loss, best))
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
    best: object = None            # BestSoFar or None
    trainer_state: object = None   # whatever else the trainer carries (auxetic: its history)


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
class TrainingLossBelow:
    """Stop rule: the training loss has fallen below `threshold`."""
    threshold: float

    def reached(self, first, now):
        return bool(np.asarray(now) < self.threshold)


@dataclass(frozen=True)
class SubtaskLossRatioBelow:
    """Stop rule: every subtask's loss has fallen below `threshold` times its first value."""
    threshold: float

    def reached(self, first, now):
        """first, now: per-subtask losses at step 0 and at the current step."""
        return bool(np.all(np.asarray(now) / np.asarray(first) < self.threshold))


def run(record, trainer, *, hparams, critical_keys, code_files, target_steps, overwrite,
        stop_rule, on_mismatch, done=None, steps_key='n_training_steps'):
    """Take `record`'s realization from wherever it stands to `target_steps` steps (or the
    stop rule). on_mismatch(record) is called when the recorded critical hyperparameters
    differ from `hparams` and overwrite is False: it may raise, else its return value ends
    the invocation as Outcome('refused'). done(record), when given, ends an invocation as
    Outcome('complete') before anything is written. steps_key: the hparams key recording
    the target step count."""
    run_dir = record.run_dir
    record.start()
    meta_file = record.meta_file_name('training_meta')

    stage = None
    if has_critical_mismatch(run_dir, hparams, filename=meta_file, critical_keys=critical_keys):
        if not overwrite:
            return Outcome('refused', refusal=on_mismatch(record))
        print(f"  --overwrite set: recorded hyperparameters differ from this run's — wiping "
              f"{run_dir} and restarting from scratch under the new hyperparameters.")
        record.wipe()
        stage = 'overwrite'
    if done is not None and done(record):
        print("Job already completed! Skipping...")
        return Outcome('complete')
    previous_target = record.config.get(steps_key) if record.has('config') else None

    trainer.prepare(record)

    resume = record.resume_state()
    if resume is None:
        state = trainer.fresh_state()
        stage = stage or 'fresh'
    elif hasattr(trainer, 'resumed_state'):
        state = trainer.resumed_state(resume)
    else:
        state = TrainState(resume.step, resume.stiffnesses, resume.loss)
    if not hasattr(trainer, 'resumed_state'):
        best = record.best_so_far if record.has('best_so_far') else None
        state = TrainState(state.step, state.stiffnesses, state.loss, best, state.trainer_state)
        if best is not None:
            print(f"  Best state    : combined_mse={best.training_loss:.4e} (loaded from disk)")

    if hasattr(trainer, 'extra_hparams'):
        hparams = {**hparams, **trainer.extra_hparams(state)}
    save_training_meta(run_dir, hparams, filename=meta_file, overwrite=overwrite,
                       critical_keys=critical_keys)
    save_run_provenance(run_dir, extra=hparams, filename=record.meta_file_name('run_provenance'))
    save_code_snapshot(run_dir, code_files)
    record.refresh()

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

    if hasattr(trainer, 'save_final'):
        trainer.save_final(record, state, status)
    else:
        record.save_final(state.stiffnesses, state.loss, state.best)
    return Outcome(stage, status, state)
