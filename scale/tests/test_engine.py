"""Fast checks for the geometry and timing math.  Run: python -m unittest discover tests"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scale_engine.spec import load_episode                      # noqa: E402
from scale_engine.timeline import Camera, captions, format_scale, layout  # noqa: E402

EP = Path(__file__).resolve().parent.parent / "episodes" / "ep001_the_eye.yaml"


class Geometry(unittest.TestCase):
    def setUp(self):
        self.ep = load_episode(EP)

    def test_integer_z_frames_each_layer_exactly(self):
        for i, layer in enumerate(self.ep.layers):
            v, cx, cy = self.ep.view(i)
            self.assertAlmostEqual(v, layer.width, delta=layer.width * 1e-9)
            self.assertAlmostEqual(cx, layer.center[0], delta=layer.width * 1e-9)
            self.assertAlmostEqual(cy, layer.center[1], delta=layer.width * 1e-9)

    def test_view_is_continuous_and_monotonic(self):
        prev = None
        for k in range(-50, len(self.ep.layers) * 100):
            z = k / 100
            v, cx, cy = self.ep.view(z)
            if prev:
                self.assertGreater(v, prev[0])
                # center never jumps by more than the view itself
                self.assertLess(math.hypot(cx - prev[1], cy - prev[2]), v)
            prev = (v, cx, cy)

    def test_meters_readout(self):
        self.assertAlmostEqual(self.ep.meters_at(0), 0.05)
        self.assertEqual(format_scale(12_000_000)[0], "12,000 km")
        self.assertEqual(format_scale(9.4607e15 * 2.5e6)[0], "2.5 million light-years")
        self.assertEqual(format_scale(0.024)[1], "10⁻² m")


class Timing(unittest.TestCase):
    def test_camera_hits_every_landing_and_snaps_back(self):
        ep = load_episode(EP)
        durs = [2.0] * len(ep.beats)
        cues, keys, total = layout(ep, durs, [None] * len(ep.beats))
        cam = Camera(keys)
        for k in keys:
            self.assertAlmostEqual(cam(k.t), k.z, places=6)
        self.assertGreater(total, sum(durs))
        snap = [b for b in ep.beats if b.move == "snap"][0]
        cue = [c for c in cues if c.beat_id == snap.id][0]
        self.assertGreater(cam(cue.start - 0.01), 15)          # out at the edge of everything
        self.assertLess(cam(cue.start + snap.snap + 0.01), 0.1)  # back at the eye

    def test_captions_cover_speech(self):
        ep = load_episode(EP)
        cues, _, _ = layout(ep, [3.0] * len(ep.beats), [None] * len(ep.beats))
        caps = captions(cues, 4, 22)
        self.assertEqual(" ".join(c.text for c in caps), " ".join(b.text for b in ep.beats))
        for a, b in zip(caps, caps[1:]):
            self.assertLessEqual(a.end, b.start + 1e-9)


if __name__ == "__main__":
    unittest.main()
