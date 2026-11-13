"""Watch the corpus folder and sync whenever files change.

  python -m pka.watch                    # sync now, then keep watching (Ctrl+C to stop)
  python -m pka.watch --interval 20 --settle 8 --auto-summaries 30

It polls the folder every --interval seconds. A change must stay unchanged for --settle seconds
before a sync starts, so a large PDF that is still being copied isn't read half-written.
(python -m pka.app runs this for you in the background.)
"""
import argparse
import threading
import time
from pathlib import Path

from pka.config import settings
from pka.ingestion.loaders import discover


def snapshot(corpus_dir) -> dict:
    """{path: (size, modified time)} of every supported file."""
    out = {}
    for p in discover(Path(corpus_dir)):
        try:
            st = p.stat()
            out[str(p)] = (st.st_size, st.st_mtime_ns)
        except OSError:
            pass                      # vanished between listing and reading
    return out


class Watcher:
    def __init__(self, sync_fn, corpus_dir=None, interval=20.0, settle=8.0, backoff=60.0,
                 log=print, clock=time.monotonic, snapshot_fn=None):
        self.sync_fn = sync_fn
        self.corpus_dir = Path(corpus_dir or settings.corpus_dir)
        self.interval, self.settle, self.backoff = interval, settle, backoff
        self.log, self.clock = log, clock
        self._snapshot = snapshot_fn or (lambda: snapshot(self.corpus_dir))
        self.last_synced = None       # snapshot that the index reflects
        self._pending = None          # (snapshot, time first seen)
        self._retry_at = 0.0

    def sync_now(self):
        snap = self._snapshot()
        self.sync_fn()
        self.last_synced, self._pending = snap, None

    def tick(self) -> str:
        """One polling step. Returns idle | waiting | synced | backoff | busy | error."""
        now = self.clock()
        if now < self._retry_at:
            return "backoff"
        snap = self._snapshot()
        if snap == self.last_synced:
            self._pending = None
            return "idle"
        if self._pending is None or self._pending[0] != snap:
            self._pending = (snap, now)            # changed (again): restart the settle timer
            return "waiting"
        if now - self._pending[1] < self.settle:
            return "waiting"
        try:
            self.sync_fn()
        except Exception as e:
            from pka.sync import SyncBusy
            if isinstance(e, SyncBusy):
                return "busy"
            self.log(f"Sync failed: {e}  (will retry in {self.backoff:.0f}s)")
            self._retry_at = now + self.backoff
            return "error"
        self.last_synced, self._pending = snap, None
        return "synced"

    def run(self, stop: threading.Event = None, initial_sync=True):
        stop = stop or threading.Event()
        if initial_sync:
            try:
                self.sync_now()
            except Exception as e:
                self.log(f"Initial sync failed: {e}  (will retry in {self.backoff:.0f}s)")
                self._retry_at = self.clock() + self.backoff
        while not stop.is_set():
            self.tick()
            stop.wait(self.interval)


def main():
    ap = argparse.ArgumentParser(description="Sync the index whenever the corpus folder changes.")
    ap.add_argument("--interval", type=float, default=20, help="seconds between checks (default 20)")
    ap.add_argument("--settle", type=float, default=8, help="seconds a change must stay quiet (default 8)")
    ap.add_argument("--auto-summaries", type=int, metavar="N",
                    help="also build summaries after each sync, using at most N API calls")
    args = ap.parse_args()

    from pka.sync import Syncer
    syncer = Syncer()

    def do_sync():
        report = syncer.sync(summaries_max_calls=args.auto_summaries)
        print(time.strftime("[%H:%M:%S] ") + report.text())

    print(f"Watching {settings.corpus_dir} (Ctrl+C to stop)")
    try:
        Watcher(do_sync, interval=args.interval, settle=args.settle).run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
