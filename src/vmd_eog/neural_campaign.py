"""New campaign dispatch; readiness never masquerades as accuracy qualification."""


def execute(stage, input_root, output, config, profile, experiment):
    if stage == "research-ready":
        from .readiness import run_readiness
        return run_readiness(input_root, output, config)
    if stage == "neural-fixture":
        from .neural_experiments import run_fixture
        return run_fixture(output, config)
    if stage == "neural-diagnosis":
        from .neural_diagnosis import run_diagnosis
        return run_diagnosis(input_root, output, config, experiment)
    if stage == "autovmd-cache":
        from .autovmd import build_cache
        return build_cache(input_root, output, config, experiment)
    if stage in ("router-train", "student-paired"):
        from .neural_experiments import train_experiment
        return train_experiment(input_root, output, config, profile, experiment)
    if stage == "autovmd-search":
        from .neural_experiments import select_search
        return select_search(input_root, output, config, experiment)
    raise NotImplementedError("Numerical implementation/qualification pending for " + stage)
