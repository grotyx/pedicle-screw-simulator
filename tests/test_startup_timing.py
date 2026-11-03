"""Startup clocks use render completion, never an event-loop timer proxy."""

import logging


def test_summary_waits_for_real_render_and_cleanup_and_logs_once(caplog):
    from src.core.startup_timing import StartupTiming

    now = [10.0]
    timing = StartupTiming(8.0, clock=lambda: now[0])
    with caplog.at_level(logging.INFO):
        timing.finish('imports', 9.0)
        now[0] = 12.0
        timing.finish('window_build', 10.0)
        timing.add_native_init(0.25)
        timing.mark_shown()
        assert not caplog.records
        now[0] = 13.0
        timing.render_completed()
        assert not caplog.records  # Cleanup is measured even if it finishes later.
        now[0] = 14.0
        timing.finish('purge', 12.0)
        timing.render_completed()
    summaries = [r.message for r in caplog.records if r.message.startswith('Startup:')]
    assert len(summaries) == 1
    assert 'imports=1.000s' in summaries[0]
    assert 'window_build=2.000s' in summaries[0]
    assert 'native_init=0.250s' in summaries[0]
    assert 'purge=2.000s' in summaries[0]
    assert 'first_render=5.000s' in summaries[0]


def test_hidden_render_does_not_count():
    from src.core.startup_timing import StartupTiming

    timing = StartupTiming(0, clock=lambda: 10)
    timing.render_completed()
    assert timing.first_render is None
    timing.mark_shown()
    timing.render_completed()
    assert timing.first_render == 10


def test_cleanup_worker_keeps_caller_free_and_is_timed():
    import threading

    from src.core.startup_timing import StartupTiming

    entered = threading.Event()
    release = threading.Event()
    timing = StartupTiming(0)

    def purge(**kwargs):
        assert kwargs == {'older_than_seconds': 3600}
        entered.set()
        assert release.wait(5)

    worker = timing.start_cleanup(purge)
    try:
        assert entered.wait(5)
        assert worker.is_alive()
        assert 'purge' not in timing.durations
    finally:
        release.set()
        worker.join(5)
    assert 'purge' in timing.durations


def test_entry_defers_purge_until_window_requests_after_show(monkeypatch):
    import main as entry
    import src.ui.main_window as window_module
    from src.core.startup_timing import set_startup_timing
    from src.core.totalseg_integration import SegmentationWorkspace

    calls = []
    monkeypatch.setattr(SegmentationWorkspace, 'purge_stale', lambda **kw: calls.append('purge'))

    def launch(*, startup, after_show):
        assert calls == []
        assert 'imports' in startup.durations
        calls.append('shown')
        worker = after_show()
        worker.join(5)
        assert calls == ['shown', 'purge']

    monkeypatch.setattr(window_module, 'main', launch)
    try:
        entry.run_application()
    finally:
        set_startup_timing(None)
