"""
Smoke test of app.py.
* If Streamlit is installed, uses Streamlit's official AppTest and clicks both buttons.
* Otherwise runs app.py against lightweight stand-ins for streamlit / plotly, which
  catches Python errors (wrong names, bad arguments) but cannot judge the visuals.
"""
import runpy, sys, types, unittest
from pathlib import Path

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Any()
    def __getattr__(self, n): return _Any()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([])


class _State(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


class _FakeSt:
    def __init__(self, pressed):
        self.pressed, self.session_state = pressed, _State()
        self.sidebar = self
        self.calls = []

    def __getattr__(self, name):                    # markdown, plotly_chart, info ... -> no-ops
        def f(*a, **k):
            self.calls.append(name)
            return _Any()
        return f

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **k):
        return value if value is not None else min_value
    number_input = slider
    def selectbox(self, label, options, index=0, **k): return list(options)[index]
    def button(self, label, **k): return any(p in label for p in self.pressed)
    def tabs(self, labels): return [self for _ in labels]
    def columns(self, spec): return [self for _ in range(spec if isinstance(spec, int) else len(spec))]
    def expander(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def progress(self, *a, **k): return _Any()
    def stop(self): raise RuntimeError("st.stop() called - input validation failed")
    def download_button(self, *a, **k): return False


class TestApp(unittest.TestCase):
    def test_app(self):
        try:
            import streamlit  # noqa
            from streamlit.testing.v1 import AppTest
            at = AppTest.from_file(APP, default_timeout=180).run()
            self.assertFalse(at.exception)
            for b in at.sidebar.button:
                b.click()
                at.run()
                self.assertFalse(at.exception, at.exception)
            return
        except ImportError:
            pass
        import numpy as np
        fake_plotly = types.ModuleType("plotly"); go = types.ModuleType("plotly.graph_objects")
        for n in ("Figure", "Scatter", "Bar", "Frame"):
            setattr(go, n, _Any)
        fake_plotly.graph_objects = go
        saved = {k: sys.modules.get(k) for k in ("streamlit", "plotly", "plotly.graph_objects")}
        for pressed in ([], ["Run Simulation"], ["Run Optimization"], ["Run Simulation", "Run Optimization"]):
            fst = _FakeSt(pressed)
            sys.modules.update({"streamlit": fst, "plotly": fake_plotly, "plotly.graph_objects": go})
            for m in [m for m in sys.modules if m.startswith("visualization")]:
                del sys.modules[m]
            try:
                runpy.run_path(APP, run_name="__main__")
            finally:
                for k, v in saved.items():
                    if v is None: sys.modules.pop(k, None)
                    else: sys.modules[k] = v
            self.assertNotIn("error", fst.calls, f"st.error called for {pressed}")


if __name__ == "__main__":
    unittest.main()
