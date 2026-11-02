"""Startup measurements shared by the entry point and native render widgets."""

import logging
import threading
import time

logger = logging.getLogger(__name__)
_active = None


class StartupTiming:
    """Log one summary once cleanup and the first visible render have finished."""

    def __init__(self, entered, *, clock=time.monotonic):
        self.entered = entered
        self.clock = clock
        self.durations = {}
        self.native_init = 0.0
        self._native_ready = False
        self.first_render = None
        self._shown = False
        self._logged = False
        self._lock = threading.Lock()

    def finish(self, phase, started):
        with self._lock:
            self.durations[phase] = self.clock() - started
            self._maybe_log()

    def add_native_init(self, duration):
        with self._lock:
            self.native_init += duration
            self._native_ready = True
            self._maybe_log()

    def mark_shown(self):
        self._shown = True

    def render_completed(self):
        with self._lock:
            if self._shown and self.first_render is None:
                self.first_render = self.clock() - self.entered
                self._maybe_log()

    def _maybe_log(self):
        if self._logged or self.first_render is None or not self._native_ready or not {'imports', 'window_build', 'purge'} <= self.durations.keys():
            return
        self._logged = True
        logger.info(
            'Startup: imports=%.3fs window_build=%.3fs native_init=%.3fs '
            'purge=%.3fs first_render=%.3fs (since process entry)',
            self.durations['imports'], self.durations['window_build'],
            self.native_init, self.durations['purge'], self.first_render,
        )

    def start_cleanup(self, purge):
        """Run potentially slow deletion outside the GUI thread after show()."""
        def clean():
            started = self.clock()
            try:
                purge(older_than_seconds=3600)
            except Exception:
                logger.exception('Stale segmentation workspace cleanup failed')
            finally:
                self.finish('purge', started)

        worker = threading.Thread(target=clean, name='startup-cleanup', daemon=True)
        worker.start()
        return worker


def set_startup_timing(timing):
    global _active
    _active = timing


def startup_timing():
    return _active
