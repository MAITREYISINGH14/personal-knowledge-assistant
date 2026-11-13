"""Run the assistant: web interface + automatic folder syncing, in one command.

  python -m pka.app                         # open http://127.0.0.1:8000
  python -m pka.app --port 9000 --no-browser
  python -m pka.app --no-watch              # don't sync automatically (the Sync button still works)
  python -m pka.app --auto-summaries 30     # also build summaries after syncs (max 30 API calls each)

Drop a file into the corpus folder and, within about half a minute, you can ask about it.
"""
import argparse
import threading
import time
import webbrowser

from pka.config import settings


def main():
    ap = argparse.ArgumentParser(description="Personal Knowledge Assistant")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--allow-network", action="store_true",
                    help="allow other computers to connect (this app has no login!)")
    ap.add_argument("--no-watch", action="store_true", help="don't sync the folder automatically")
    ap.add_argument("--interval", type=float, default=20, help="seconds between folder checks")
    ap.add_argument("--settle", type=float, default=8, help="seconds a change must stay quiet before syncing")
    ap.add_argument("--auto-summaries", type=int, metavar="N",
                    help="build summaries after each sync, using at most N API calls")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_network:
        raise SystemExit("Refusing to listen on a network address: this app has no login, so anyone on "
                         "the network could read your documents. Use --allow-network if you understand that.")

    import uvicorn
    from pka.api.main import create_app
    from pka.service import Assistant
    from pka.watch import Watcher

    assistant = Assistant(auto_summaries=args.auto_summaries)
    app = create_app(assistant)
    stop = threading.Event()

    def sync_and_log():
        report = assistant.run_sync()
        if report.changed or report.failed or report.warnings:
            print(time.strftime("[%H:%M:%S] ") + report.text(), flush=True)
        return report

    def background():
        assistant.warm_up()
        if args.no_watch:
            return
        assistant.watching = True
        Watcher(sync_and_log, interval=args.interval, settle=args.settle,
                log=lambda m: print(m, flush=True)).run(stop)

    threading.Thread(target=background, daemon=True, name="pka-background").start()

    url = f"http://{'127.0.0.1' if args.host in ('localhost', '::1') else args.host}:{args.port}"
    print(f"Personal Knowledge Assistant at {url}   (Ctrl+C to stop)")
    print(f"Watching {settings.corpus_dir}" if not args.no_watch else "Automatic syncing is off.")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
