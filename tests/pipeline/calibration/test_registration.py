"""Guards against the exact failure mode that let swe-smith sit silently unwired in
`validate-graders` for a while: a source listed in `config/calibration.yaml`'s `gradeable_sources`
but missing from one of the dispatch tables a real calibration run actually depends on. Written
generically against the real config file (not a hard-coded source list), so it stays meaningful as
sources are added or removed."""
from router.common.config import load_calibration_config
from router.pipeline.calibration import calibrate as calibrate_mod
from router.pipeline.calibration import runner as runner_mod
from router.pipeline.calibration import tasks as tasks_mod


def test_every_gradeable_source_has_a_task_loader():
    calibration_config = load_calibration_config()
    for source in calibration_config.gradeable_sources:
        assert source in tasks_mod._LOADERS, f"{source!r} has no entry in tasks._LOADERS"


def test_every_gradeable_source_has_a_grader():
    calibration_config = load_calibration_config()
    for source in calibration_config.gradeable_sources:
        assert source in calibrate_mod._GRADERS, f"{source!r} has no entry in calibrate._GRADERS"


def test_every_gradeable_source_has_a_prompt_instruction():
    calibration_config = load_calibration_config()
    for source in calibration_config.gradeable_sources:
        assert source in runner_mod._INSTRUCTIONS, f"{source!r} has no entry in runner._INSTRUCTIONS"


def test_every_gradeable_source_resolves_a_callable_reference_and_null_grader():
    # Resolution itself must not raise a KeyError for a registered source — a source with no
    # special-cased reference/null grader must at least have a generic entry in _GRADERS, which
    # grade_reference/grade_null fall back to. This is the "unwired source raises loudly" guarantee
    # the dispatch collapse exists to provide.
    calibration_config = load_calibration_config()
    for source in calibration_config.gradeable_sources:
        if source not in calibrate_mod._REFERENCE_GRADERS:
            assert source in calibrate_mod._GRADERS, f"{source!r} has no reference grader of any kind"
        if source not in calibrate_mod._NULL_GRADERS:
            assert source in calibrate_mod._GRADERS, f"{source!r} has no null grader of any kind"


def test_reference_and_null_grader_dicts_have_identical_keys_all_present_in_graders():
    assert set(calibrate_mod._REFERENCE_GRADERS) == set(calibrate_mod._NULL_GRADERS)
    for source in calibrate_mod._REFERENCE_GRADERS:
        assert source in calibrate_mod._GRADERS
