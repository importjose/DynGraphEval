"""CPU-only regression test for time-scale job storage isolation."""

import ast
import importlib.util
import json
import math
import textwrap
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class TimeScaleIsolationTest(unittest.TestCase):
    def test_resume_rejects_wrong_experiment_before_loading_weights(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("tgn", "tgat", "graphmixer"):
            source = (root / "models" / f"{name}_time_scale" / "patches" /
                      "train_link_prediction.py").read_text()
            tree = ast.parse(source)
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == "_load_resume_checkpoint")
            namespace = {
                "os": os,
                "torch": types.SimpleNamespace(load=lambda *args, **kwargs: {
                    "experiment": {"variant": "wrong_variant", "run_id": "old_job",
                                   "time_scale": "1.0", "model_name": "TGN"}}),
            }
            exec(compile(ast.Module(body=[function], type_ignores=[]), "resume", "exec"), namespace)
            model = types.SimpleNamespace(load_state_dict=lambda _: self.fail("Loaded wrong weights"))
            env = {"TIME_ENCODER_VARIANT": f"{name}_bin_day", "TIME_ENCODER_RUN_ID": "new_job",
                   "TIME_SCALE": "1.1574074074074073e-05", "TIME_BIN_WIDTH": "86400.0",
                   "TIME_ENCODING_MODE": "binned"}
            with patch.dict(os.environ, env):
                with self.assertRaisesRegex(ValueError, "experiment mismatch"):
                    namespace["_load_resume_checkpoint"]("old.pkl", model, None, None,
                                                         name.upper(), "cpu")

    def test_binning_differs_from_continuous_scaling(self):
        for width in (3600, 86400, 604800, 2592000):
            self.assertEqual(math.floor(0 / width), 0)
            self.assertEqual(math.floor((width - 1) / width), 0)
            self.assertEqual(math.floor(width / width), 1)
            self.assertEqual(math.floor((width + 1) / width), 1)
            self.assertNotEqual((width + 1) / width, 1)
        # The training patch must apply before model construction, for all architectures.
        root = Path(__file__).resolve().parents[1]
        for model in ("tgn", "tgat", "graphmixer"):
            source = (root / "models" / f"{model}_time_scale" / "patches" /
                      "train_link_prediction.py").read_text()
            self.assertLess(source.index("[time-bin] Applying floor(age"),
                            source.index("if args.model_name == 'TGAT':"))
            self.assertIn("torch.floor(ages / _width)", source)

    def test_training_patch_buckets_ages_in_all_three_models(self):
        root = Path(__file__).resolve().parents[1]
        for model in ("tgn", "tgat", "graphmixer"):
            source = (root / "models" / f"{model}_time_scale" / "patches" /
                      "train_link_prediction.py").read_text()
            start = source.index("        # Patch the shared encoder before constructing ANY backbone.")
            end = source.index("        # ── Build model", start)
            snippet = textwrap.dedent(source[start:end]).replace(
                "from models.modules import TimeEncoder", "# supplied by test")
            class TimeEncoder:
                def forward(self, timestamps):
                    return timestamps
            env = {"TIME_ENCODING_MODE": "binned", "TIME_BIN_WIDTH": "3600",
                   "TIME_SCALE": str(1 / 3600)}
            namespace = {"os": os, "torch": types.SimpleNamespace(floor=math.floor),
                         "np": types.SimpleNamespace(isfinite=math.isfinite),
                         "TimeEncoder": TimeEncoder}
            with patch.dict(os.environ, env):
                exec(snippet, namespace)
            encoder = TimeEncoder()
            self.assertEqual([encoder.forward(age) for age in (0, 3599, 3600, 3601, 7200)],
                             [0, 0, 1, 1, 2])

    def test_evaluation_uses_same_bins_as_training(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "hpc" / "eval.py").read_text()
        for name in ("tgat_bin_hour", "tgat_bin_month"):
            self.assertIn(f'"{name}"', source)
        start = source.index("    _TIME_GRAN_WIDTHS =")
        end = source.index("    # ── Load dataset", start)
        snippet = textwrap.dedent(source[start:end])
        for model in ("tgn_bin_hour", "tgat_bin_hour", "tgat_bin_day",
                      "tgat_bin_week", "tgat_bin_month", "graphmixer_bin_week"):
            class TimeEncoder:
                def forward(self, timestamps):
                    return timestamps
            modules = {key: types.ModuleType(key) for key in
                       ("models", "models.tgn", "models.tgn.tpnet_components")}
            modules["models.tgn.tpnet_components"].TimeEncoder = TimeEncoder
            namespace = {"torch": types.SimpleNamespace(floor=math.floor),
                         "args": types.SimpleNamespace(model=model), "_log": lambda _: None}
            with patch.dict(sys.modules, modules):
                exec(snippet, namespace)
            width = namespace["_TIME_GRAN_WIDTHS"][model.rsplit("_", 1)[-1]]
            self.assertEqual([TimeEncoder().forward(age) for age in (0, width-1, width, width+1)],
                             [0, 0, 1, 1])

    def test_variants_and_jobs_do_not_share_models_or_outputs(self):
        root = Path(__file__).resolve().parents[1]
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True, get_device_name=lambda gpu: "test GPU"))
        variants = [
            ("tgn_time_scale", "hour", "TGN"),
            ("tgn_time_scale", "day", "TGN"),
            ("tgat_time_scale", "hour", "TGAT"),
            ("tgat_time_scale", "day", "TGAT"),
            ("tgat_time_scale", "month", "TGAT"),
            ("graphmixer_time_scale", "hour", "GraphMixer"),
        ]
        seen_directories = set()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"torch": fake_torch}):
            cache = Path(tmp) / "storage" / "datasets"
            cache.mkdir(parents=True)
            for job in ("job101", "job202"):
                for script, granularity, model_name in variants:
                    module_path = root / "models" / script / "train.py"
                    spec = importlib.util.spec_from_file_location(f"test_{script}", module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # Slurm reuses one temporary repo for multiple granularities.
                    repo = Path(tmp) / f"repo_{script}_{job}"
                    repo.mkdir(exist_ok=True)
                    variant_name = f"{script.removesuffix('_time_scale')}_bin_{granularity}"
                    output = (Path(tmp) / "storage" / "runs" / "tgbl-wiki" /
                              variant_name / "seed0" / job)

                    def pretend_train(command, extra_env):
                        self.assertEqual(extra_env["TIME_ENCODER_VARIANT"], variant_name)
                        self.assertEqual(extra_env["TIME_ENCODER_RUN_ID"], job)
                        self.assertEqual(extra_env["TIME_ENCODING_MODE"], "binned")
                        self.assertIn("TIME_BIN_WIDTH", extra_env)
                        isolated = (repo / "saved_models").resolve()
                        self.assertNotIn(isolated, seen_directories)
                        seen_directories.add(isolated)
                        self.assertEqual(isolated, (output / "trainer").resolve())
                        (isolated / f"run0_link_tgbl-wiki_{model_name}_seed0.pkl").write_text(job)

                    argv = [str(root / "models" / script / "train.py"),
                            "--granularity", granularity, "--run_id", job,
                            "--repo_dir", str(repo), "--datasets_cache", str(cache),
                            "--run_dir", str(output)]
                    with patch.object(sys, "argv", argv), \
                         patch.object(module, "_apply_patches"), \
                         patch.object(module, "run", side_effect=pretend_train):
                        module.main()
                    self.assertEqual((output / "checkpoint.pkl").read_text(), job)
                    manifest = json.loads((output / "manifest.json").read_text())
                    self.assertEqual(manifest["variant"], variant_name)
                    self.assertEqual(manifest["run_id"], job)
                    self.assertEqual(manifest["encoding_mode"], "binned")
            self.assertEqual(len(seen_directories), len(variants) * 2)


if __name__ == "__main__":
    unittest.main()
