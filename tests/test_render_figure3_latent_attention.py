import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT = PROJECT_ROOT / "render_figure3_latent_attention.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Figure3LatentAttentionRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.render_mod = load_module("render_figure3_latent_attention_test", RENDER_SCRIPT)

    def test_renderer_fails_clearly_when_required_source_table_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source_tables").mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "required source table"):
                self.render_mod.load_source_tables(root)


if __name__ == "__main__":
    unittest.main()
