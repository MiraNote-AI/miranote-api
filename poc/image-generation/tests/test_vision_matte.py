import io
import os
import sys
import threading
import unittest
from unittest import mock

from PIL import Image

import config
from cutout import vision_matte


def rgba_png(size=(100, 100), box=(25, 25, 75, 75)) -> bytes:
    """A matte with one opaque rectangle, for the box-clipping assertions."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    img.paste((200, 120, 60, 255), box)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class AvailabilityTests(unittest.TestCase):
    """The Vision modes are macOS-only and need a compiled Swift helper.

    available() is what lifespan consults to decide whether to start the helper,
    so a wrong answer either crashes a Linux boot or hides a missing build.
    """

    def test_needs_darwin(self):
        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch.object(os.path, "exists", return_value=True):
            self.assertFalse(vision_matte.available())

    def test_needs_the_binary(self):
        with mock.patch.object(sys, "platform", "darwin"), \
             mock.patch.object(os.path, "exists", return_value=False):
            self.assertFalse(vision_matte.available())

    def test_darwin_with_binary(self):
        with mock.patch.object(sys, "platform", "darwin"), \
             mock.patch.object(os.path, "exists", return_value=True):
            self.assertTrue(vision_matte.available())

    def test_start_refuses_when_unavailable(self):
        """Rather than falling back to rembg -- a Vision run must be a Vision run."""
        with mock.patch.object(vision_matte, "available", return_value=False):
            with self.assertRaises(vision_matte.VisionMatteUnavailable) as caught:
                vision_matte._start()
        self.assertIn("vision matte unavailable", str(caught.exception))
        self.assertIn("swiftc", str(caught.exception),
                      "the error should say how to build the helper")

    def test_binary_path_comes_from_config(self):
        self.assertIn(config.VISION_MATTE_BIN.split("/")[-1], vision_matte.binary_path())


class FakeHelper:
    """A stand-in for the Swift process: answers, crashes, or hangs.

    `reply` is the line it writes back; None means it never answers, which is
    the case the watchdog exists for. Killing it makes readline return "",
    exactly as the real pipe does when the process dies.
    """

    def __init__(self, reply="", hang=False):
        self._reply = reply
        self._hang = hang
        self._killed = threading.Event()
        self.kill_count = 0
        self.stdin = mock.Mock()
        self.stdout = mock.Mock()
        self.stdout.readline = self._readline

    def _readline(self):
        if self._hang:
            # Blocks like the real readline does, and ends the same way: the
            # pipe closes when the process is killed, so the read returns "".
            self._killed.wait(timeout=10)
            return ""
        return self._reply

    def kill(self):
        self.kill_count += 1
        self._killed.set()

    def poll(self):
        return -9 if self._killed.is_set() else None


class TimeoutTests(unittest.TestCase):
    """The helper is held behind one lock for a whole round trip, so a hang
    would stall not just its own request but every cutout queued behind it,
    forever. These pin the bound that turns that into one failed request."""

    def setUp(self):
        vision_matte._reset()
        self.addCleanup(vision_matte._reset)

    def test_hanging_helper_is_killed_and_raises(self):
        helper = FakeHelper(hang=True)
        with mock.patch.object(vision_matte, "_helper", return_value=helper), \
             mock.patch.object(config, "VISION_MATTE_TIMEOUT", 0.05):
            with self.assertRaises(vision_matte.VisionMatteUnavailable) as caught:
                vision_matte.remove_background(b"raw")
        self.assertEqual(helper.kill_count, 1, "the watchdog must kill the helper")
        self.assertIn("stopped answering", str(caught.exception),
                      "a hang must not be reported as an ordinary crash")

    def test_the_lock_is_free_afterwards(self):
        """The whole point: the next request must not inherit the hang."""
        helper = FakeHelper(hang=True)
        with mock.patch.object(vision_matte, "_helper", return_value=helper), \
             mock.patch.object(config, "VISION_MATTE_TIMEOUT", 0.05):
            with self.assertRaises(vision_matte.VisionMatteUnavailable):
                vision_matte.remove_background(b"raw")
        self.assertTrue(vision_matte._lock.acquire(timeout=1))
        vision_matte._lock.release()

    def test_a_crash_is_not_reported_as_a_timeout(self):
        helper = FakeHelper(reply="")            # immediate EOF
        with mock.patch.object(vision_matte, "_helper", return_value=helper):
            with self.assertRaises(vision_matte.VisionMatteUnavailable) as caught:
                vision_matte.remove_background(b"raw")
        self.assertIn("exited mid-request", str(caught.exception))

    def test_a_prompt_reply_does_not_trip_the_watchdog(self):
        helper = FakeHelper(reply='{"ok":false,"error":"no foreground instances"}\n')
        with mock.patch.object(vision_matte, "_helper", return_value=helper):
            with self.assertRaises(vision_matte.VisionMatteUnavailable) as caught:
                vision_matte.remove_background(b"raw")
        self.assertEqual(helper.kill_count, 0)
        self.assertIn("no foreground instances", str(caught.exception))

    def test_unreadable_reply_is_not_a_bare_500(self):
        """A JSONDecodeError would escape main.py's except and read as a bug in
        the cutout rather than "the helper said something unexpected"."""
        helper = FakeHelper(reply="not json at all\n")
        with mock.patch.object(vision_matte, "_helper", return_value=helper):
            with self.assertRaises(vision_matte.VisionMatteUnavailable) as caught:
                vision_matte.remove_background(b"raw")
        self.assertIn("unreadable reply", str(caught.exception))


class ShutdownTests(unittest.TestCase):
    """The helper is a child of the server process. Nothing else reaps it, so
    without this it outlives every stop and every --reload with its model
    still resident."""

    def setUp(self):
        vision_matte._reset()
        self.addCleanup(vision_matte._reset)

    def test_shutdown_kills_a_running_helper(self):
        helper = FakeHelper()
        vision_matte._process = helper
        vision_matte.shutdown()
        self.assertEqual(helper.kill_count, 1)
        self.assertIsNone(vision_matte._process)

    def test_shutdown_is_safe_when_never_started(self):
        vision_matte.shutdown()          # must not raise

    def test_shutdown_does_not_wait_on_the_lock(self):
        """Shutting down behind an in-flight matte would hang on exactly the
        case the watchdog exists for."""
        helper = FakeHelper()
        vision_matte._process = helper
        with vision_matte._lock:
            vision_matte.shutdown()
        self.assertEqual(helper.kill_count, 1)


class ModeRegistryTests(unittest.TestCase):
    """The Vision work was added as a new mode so the rembg path stays as a
    reference to A/B against. These pin that the old modes were not disturbed."""

    def setUp(self):
        # Imported lazily: main.py pulls in torch and the rembg session at import
        # time, which the other tests in this suite deliberately avoid paying for.
        import main
        self.main = main

    def test_existing_modes_are_untouched(self):
        self.assertIn("hybrid_sam_union", self.main.PROMPT_CUTOUT_MODES)
        self.assertIn("hybrid_sam_prebg_gray", self.main.PROMPT_CUTOUT_MODES)
        self.assertIs(self.main.PROMPT_CUTOUT_MODES["hybrid_sam_union"],
                      self.main._cutout_via_hybrid_sam)
        self.assertIs(self.main.PROMPT_CUTOUT_MODES["hybrid_sam_prebg_gray"],
                      self.main._cutout_via_hybrid_prebg)

    def test_default_mode_is_vision(self):
        self.assertEqual(config.DEFAULT_PROMPT_CUTOUT_MODE, "hybrid_sam_prebg_vision")

    def test_endpoint_reads_the_default_straight_from_config(self):
        """The default is the constant, not something resolved at startup.

        There used to be a _resolve_default_prompt_cutout_mode that swapped in
        DEFAULT_PROMPT_CUTOUT_MODE_NO_VISION whenever the helper was missing.
        A machine that could not build the helper then served rembg at 12-80 s
        per image with nothing in the response or the mode name saying so. The
        rollback is a config edit now, so this pins that no indirection crept
        back in between the constant and the endpoint.
        """
        import inspect
        source = inspect.getsource(self.main.cutout_image)
        self.assertIn("config.DEFAULT_PROMPT_CUTOUT_MODE", source)
        self.assertFalse(hasattr(self.main, "_resolve_default_prompt_cutout_mode"))
        self.assertFalse(hasattr(config, "DEFAULT_PROMPT_CUTOUT_MODE_NO_VISION"))

    def test_the_vision_mode_is_marked_as_needing_the_helper(self):
        """Named or defaulted to, it is served or refused -- never substituted,
        otherwise an A/B against the rembg mode could silently be rembg vs rembg."""
        self.assertEqual(self.main.VISION_PROMPT_CUTOUT_MODES,
                         {"hybrid_sam_prebg_vision"})
        self.assertNotIn("hybrid_sam_prebg_gray",
                         self.main.VISION_PROMPT_CUTOUT_MODES)

    def test_vision_mode_is_registered(self):
        self.assertIs(self.main.PROMPT_CUTOUT_MODES["hybrid_sam_prebg_vision"],
                      self.main._cutout_via_hybrid_prebg_vision)


class AutoModeDefaultTests(unittest.TestCase):
    """The no-prompt path, where the whole foreground is the subject.

    Same shape as the prompted default, and the same Vision-only policy -- with
    nothing here to hide a substitution behind DINO and SAM, this is the branch
    where a silent step-down was most visible as "the app just got 100x slower".
    """

    def setUp(self):
        import main
        self.main = main

    def test_default_auto_mode_is_vision(self):
        self.assertEqual(config.DEFAULT_AUTO_CUTOUT_MODE, "vision")

    def test_endpoint_reads_the_default_straight_from_config(self):
        """See the prompted-path twin: no startup step-down, config is the switch."""
        import inspect
        source = inspect.getsource(self.main.cutout_image)
        self.assertIn("config.DEFAULT_AUTO_CUTOUT_MODE", source)
        self.assertFalse(hasattr(self.main, "_resolve_default_auto_cutout_mode"))
        self.assertFalse(hasattr(config, "DEFAULT_AUTO_CUTOUT_MODE_NO_VISION"))

    def test_the_vision_mode_is_marked_as_needing_the_helper(self):
        """?mode=vision stays a 503 on a helper-less machine, never becomes rembg."""
        self.assertEqual(self.main.VISION_AUTO_CUTOUT_MODES, {"vision"})
        self.assertNotIn("auto", self.main.VISION_AUTO_CUTOUT_MODES)

    def test_rembg_is_still_reachable_by_name(self):
        """Flipping the default only matters if the old path is still an A/B
        baseline a caller can name."""
        self.assertEqual(self.main.AUTO_CUTOUT_MODES, {"vision", "auto"})

    def test_prompt_and_auto_mode_names_are_disjoint(self):
        """One `mode` param serves both branches, so an overlap would make the
        same value mean two different pipelines depending on `prompt`."""
        self.assertFalse(
            self.main.AUTO_CUTOUT_MODES & set(self.main.PROMPT_CUTOUT_MODES))


class PrebgLabelTests(unittest.IsolatedAsyncioTestCase):
    """extras["prebg"] is how a stored result says which background remover ran;
    without it the two prebg modes produce indistinguishable output."""

    async def asyncSetUp(self):
        import main
        self.main = main

    async def _run(self, mode_fn, matte_marker):
        timings = {}
        with mock.patch.object(self.main, "_flatten_on_bg", return_value=b"flat"), \
             mock.patch.object(self.main, "_cutout_via_hybrid_sam",
                               return_value=(b"png", (0, 0, 1, 1), {})) as sam:
            out, bbox, extras = await mode_fn(b"raw", "a cat", timings)
        sam.assert_awaited_once()
        self.assertEqual(timings_keys := set(timings), {matte_marker, "flatten"},
                         f"unexpected stages {timings_keys}")
        return extras

    async def test_rembg_mode_labels_gray(self):
        with mock.patch.object(self.main, "_rembg_matte", return_value=b"cut"):
            extras = await self._run(self.main._cutout_via_hybrid_prebg, "rembg")
        self.assertEqual(extras["prebg"], "gray")

    async def test_vision_mode_labels_vision(self):
        # A real PNG, not a sentinel: the Vision mode reads the matte's alpha to
        # clip the detector box, so it must be decodable.
        with mock.patch.object(vision_matte, "remove_background",
                               return_value=rgba_png()):
            extras = await self._run(self.main._cutout_via_hybrid_prebg_vision,
                                     "vision_matte")
        self.assertEqual(extras["prebg"], "vision")
        self.assertEqual(extras["matte_bbox"], [250.0, 250.0, 750.0, 750.0])

    async def test_vision_mode_propagates_failure(self):
        """No silent fallback: a broken helper must surface, not quietly become rembg."""
        with mock.patch.object(
                vision_matte, "remove_background",
                side_effect=vision_matte.VisionMatteUnavailable("helper exited")):
            with self.assertRaises(vision_matte.VisionMatteUnavailable):
                await self.main._cutout_via_hybrid_prebg_vision(b"raw", "a cat", {})


class MatteClipTests(unittest.TestCase):
    """Clipping the detector box to the matte's extent.

    A box covering most of the frame gives SAM almost nothing to go on, and on a
    prebg image it answers "the flat background" -- on 06_parfait all three
    candidates missed the subject entirely. Outside the matte is background by
    construction, so this can only tighten the box.
    """

    def setUp(self):
        import main
        self.main = main

    def test_alpha_bbox_is_normalised_to_1000(self):
        self.assertEqual(self.main._alpha_bbox(rgba_png()),
                         (250.0, 250.0, 750.0, 750.0))

    def test_alpha_bbox_none_for_fully_transparent(self):
        self.assertIsNone(self.main._alpha_bbox(rgba_png(box=(0, 0, 0, 0))))

    def test_intersect_tightens(self):
        loose = (0.0, 298.0, 1000.0, 999.0)     # the Gemini box that failed
        matte = (55.0, 414.0, 984.0, 915.0)
        self.assertEqual(self.main._intersect(loose, matte), matte)

    def test_intersect_is_none_when_disjoint(self):
        self.assertIsNone(self.main._intersect((0, 0, 100, 100), (200, 200, 300, 300)))

    def test_rembg_mode_does_not_clip(self):
        """The rembg prebg mode is the untouched baseline the Vision mode is
        A/B'd against, so it must keep taking the detector box as-is."""
        import inspect
        source = inspect.getsource(self.main._cutout_via_hybrid_prebg)
        self.assertNotIn("clip_to_matte", source)


class StickerMatteTests(unittest.TestCase):
    """/generate's sticker cutout, the last rembg caller to move to Vision.

    It was also the last place a fallback survived: a Vision failure here used
    to re-cut with rembg, on the reasoning that the image was already generated
    and paid for. That is gone -- see test_vision_failure_raises.
    """

    def setUp(self):
        import main
        self.main = main

    def test_default_sticker_matte_is_vision(self):
        self.assertEqual(config.DEFAULT_STICKER_MATTE, "vision")

    def test_endpoint_reads_the_default_straight_from_config(self):
        import inspect
        source = inspect.getsource(self.main._generate)
        self.assertIn("config.DEFAULT_STICKER_MATTE", source)
        self.assertFalse(hasattr(self.main, "_resolve_default_sticker_matte"))
        self.assertFalse(hasattr(config, "DEFAULT_STICKER_MATTE_NO_VISION"))

    def test_rembg_is_still_reachable_by_name(self):
        self.assertEqual(self.main.STICKER_MATTE_MODES, {"vision", "rembg", "none"})

    def test_none_skips_the_matte_entirely(self):
        """The iOS app mattes on its own device, so it asks for the sticker raw.

        `none` must reach neither remover: running one and discarding it would
        pay the Vision queue for nothing, and running rembg would hand back an
        image the app then mattes a second time.
        """
        import inspect
        source = inspect.getsource(self.main._generate)
        self.assertIn('matte != "none"', source,
                      "removal must be skipped for none, not just validated")
        self.assertIn("mattable", source)

    def test_none_is_validated_like_any_other_matte(self):
        """It goes through the same 400 check, so a typo like 'non' still costs
        no image generation."""
        self.assertIn("none", self.main.STICKER_MATTE_MODES)

    def test_none_does_not_need_the_vision_helper(self):
        self.assertNotIn("none", self.main.VISION_STICKER_MATTE_MODES)

    def test_the_vision_matte_is_marked_as_needing_the_helper(self):
        self.assertEqual(self.main.VISION_STICKER_MATTE_MODES, {"vision"})
        self.assertNotIn("rembg", self.main.VISION_STICKER_MATTE_MODES)

    def test_vision_failure_raises(self):
        """No rembg re-cut, whether or not the caller named the matte.

        A sticker the caller cannot identify the remover of is worse than a
        failure they can retry: the quietly-rembg'd one was indistinguishable
        from a Vision one once it landed on a page. The discarded generation is
        the accepted cost, and the endpoint turns this into a 503.
        """
        with mock.patch.object(vision_matte, "remove_background",
                               side_effect=vision_matte.VisionMatteUnavailable("no instances")), \
             mock.patch.object(self.main, "remove", return_value=b"rembg-cut") as fallback:
            with self.assertRaises(vision_matte.VisionMatteUnavailable):
                self.main._remove_sticker_bg(b"raw", "vision")
        fallback.assert_not_called()

    def test_vision_failure_is_a_503(self):
        import inspect
        source = inspect.getsource(self.main._generate)
        self.assertIn("except vision_matte.VisionMatteUnavailable", source)
        self.assertIn("status_code=503", source)

    def test_rembg_mode_does_not_touch_vision(self):
        # _rembg is mocked too, not just remove: it is the lazy session builder,
        # and leaving it real downloads and loads a background-removal model in
        # the middle of a unit test.
        with mock.patch.object(vision_matte, "remove_background") as vision, \
             mock.patch.object(self.main, "_rembg", return_value=mock.sentinel.session), \
             mock.patch.object(self.main, "remove", return_value=b"rembg-cut") as rembg:
            out = self.main._remove_sticker_bg(b"raw", "rembg")
        self.assertEqual(out, b"rembg-cut")
        self.assertEqual(rembg.call_args.kwargs["session"], mock.sentinel.session)
        vision.assert_not_called()

    def test_rembg_is_not_loaded_until_something_needs_it(self):
        """Vision-only means nothing reaches rembg on a normal run, so paying
        its model load at every startup buys nothing but slower boots."""
        import inspect
        source = inspect.getsource(self.main.lifespan)
        self.assertNotIn("new_session", source)
        self.assertIn("if _rembg_session is None", inspect.getsource(self.main._rembg))

    def test_removal_runs_off_the_event_loop(self):
        """rembg is ~15 s per image and /generate pays it NUMBER_OF_IMAGES
        times; called inline it blocks every other request for that long."""
        import inspect
        source = inspect.getsource(self.main._generate)
        self.assertIn("asyncio.to_thread(_remove_sticker_bg", source)


if __name__ == "__main__":
    unittest.main()
