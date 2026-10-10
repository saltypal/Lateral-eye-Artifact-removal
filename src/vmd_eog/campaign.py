"""Explicit notebook stage dispatch. Each stage consumes archived artifacts."""
import tempfile
from .io import unpack_source
from .data import audit_sources


def execute(stage, input_root, output, config, profile):
    if stage == "audit":
        root = unpack_source(input_root, tempfile.mkdtemp(prefix="eog-audit-"), output)
        audit_sources(root, output, config, profile)
    elif stage == "corpus":
        from .corpus import build_corpus
        build_corpus(input_root, output, config, profile)
    elif stage == "source-fixture":
        from .corpus import source_fixture
        source_fixture(input_root,output,config,profile)
    elif stage == "classical-fixture":
        from .classical_fixture import run_fixture
        run_fixture(output,config)
    elif stage == "paper-fixture":
        from .paper_evaluation import run_fixture
        run_fixture(output)
    elif stage == "native-protocol":
        from .native_protocol import run_native_protocol
        run_native_protocol(input_root, output, config, profile)
    elif stage == "vmd-diagnosis":
        from .vmd_diagnosis import run_diagnosis
        run_diagnosis(input_root, output, config, profile)
    elif stage in ("vmd", "posterior", "regional", "review"):
        from .experiments import run_stage
        run_stage(stage, input_root, output, config, profile)
    else:
        raise NotImplementedError(f"Stage {stage} has not passed its implementation gate")
