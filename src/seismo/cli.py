"""Command line: python -m seismo <demo|replay|live|api|ui|analyze|schema>."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from seismo import __version__
from seismo.config import PACKAGE_DIR, load_settings

UI_APP = PACKAGE_DIR / "ui" / "app.py"


def _child_env(settings) -> dict[str, str]:
    env = os.environ.copy()
    src = str(settings.root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("SEISMO_ROOT", str(settings.root))
    return env


def _aggregator(settings):
    from seismo.signals.aggregator import EntityAggregator

    return EntityAggregator(
        half_life_hours=float(settings.get("aggregation.half_life_hours", 6)),
        impact_window_hours=float(settings.get("aggregation.impact_window_hours", 6)),
        history_hours=float(settings.get("aggregation.history_hours", 72)),
    )


def cmd_replay(args: argparse.Namespace) -> int:
    from seismo.engine import build_engine
    from seismo.ingest.replay import ReplayAdapter
    from seismo.runner import run_pipeline
    from seismo.store.sqlite import SQLiteStore

    settings = load_settings()
    pack = Path(args.pack) if args.pack else settings.path("replay.default_pack")
    speed = float(args.speed if args.speed is not None else settings.get("replay.speed", 600))
    store = SQLiteStore(settings.path("store.path"))
    if args.reset:
        store.reset()
    engine = build_engine(settings)
    adapter = ReplayAdapter(pack, speed=speed, max_gap_seconds=float(settings.get("replay.max_gap_seconds", 2)))
    from seismo.consumers import build_consumers

    consumers = build_consumers(settings, store)
    t0 = time.perf_counter()
    counts = asyncio.run(run_pipeline([adapter], engine, _aggregator(settings), store, consumers=consumers))
    gate = store.records("gate")
    print(
        f"Replayed {pack.name}: {counts['documents']} documents -> {counts['signals']} document signals, "
        f"{counts['events']} event updates, {len(gate)} gate decisions, {len(store.records('stress'))} stress runs, "
        f"{len(store.records('weights'))} index snapshots in {time.perf_counter() - t0:.1f}s "
        f"(sentiment backend: {engine.backend.name})"
    )
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from seismo.engine import build_engine
    from seismo.ingest.bluesky import BlueskyAdapter
    from seismo.ingest.edgar import EdgarAdapter
    from seismo.ingest.gdelt import GdeltAdapter
    from seismo.ingest.replay import Recorder
    from seismo.runner import run_pipeline
    from seismo.store.sqlite import SQLiteStore

    settings = load_settings()
    engine = build_engine(settings)
    universe = engine.universe
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    adapters = []
    if "gdelt" in sources:
        adapters.append(GdeltAdapter(
            universe,
            poll_seconds=float(settings.get("gdelt.poll_seconds")),
            timespan=str(settings.get("gdelt.timespan")),
            max_records=int(settings.get("gdelt.max_records")),
            terms_per_query=int(settings.get("gdelt.terms_per_query")),
            pause_seconds=float(settings.get("gdelt.pause_seconds")),
        ))
    if "edgar" in sources:
        adapters.append(EdgarAdapter(
            universe,
            user_agent=str(settings.get("edgar.user_agent")),
            poll_seconds=float(settings.get("edgar.poll_seconds")),
            universe_only=bool(settings.get("edgar.universe_only")),
        ))
    if "bluesky" in sources:
        adapters.append(BlueskyAdapter(
            universe,
            endpoint=str(settings.get("bluesky.endpoint")),
            max_cashtags=int(settings.get("bluesky.max_cashtags")),
            min_chars=int(settings.get("bluesky.min_chars")),
            max_posts_per_author_hour=int(settings.get("bluesky.max_posts_per_author_hour")),
        ))
    if not adapters:
        print("No sources selected. Use --sources gdelt,edgar,bluesky")
        return 2
    store = SQLiteStore(settings.path("store.path"))
    if args.reset:
        store.reset()
    recorder = None
    if args.record:
        recorder = Recorder(args.record)
        print(f"Recording every ingested document to {args.record}")
    stop_after = args.minutes * 60 if args.minutes else None
    from seismo.consumers import build_consumers

    consumers = build_consumers(settings, store)
    print(f"Streaming from {', '.join(a.name for a in adapters)}" + (f" for {args.minutes:g} min" if stop_after else "") + ". Ctrl-C to stop.")
    try:
        counts = asyncio.run(run_pipeline(adapters, engine, _aggregator(settings), store, recorder=recorder,
                                          stop_after_seconds=stop_after, consumers=consumers))
        print(f"Done: {counts['documents']} documents -> {counts['signals']} signals")
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        "seismo.api.app:app",
        host=args.host or str(settings.get("api.host")),
        port=int(args.port or settings.get("api.port")),
        log_level="info",
    )
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    settings = load_settings()
    port = str(args.port or settings.get("ui.port"))
    cmd = [sys.executable, "-m", "streamlit", "run", str(UI_APP), "--server.port", port]
    return subprocess.call(cmd, cwd=settings.root, env=_child_env(settings))


def cmd_demo(args: argparse.Namespace) -> int:
    settings = load_settings()
    env = _child_env(settings)
    env["SEISMO_AUTOSTART"] = "1"
    api_port = str(settings.get("api.port"))
    ui_port = str(settings.get("ui.port"))
    procs = []
    if not args.no_api:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "seismo.api.app:app", "--port", api_port, "--log-level", "warning"],
            cwd=settings.root, env=env,
        ))
    procs.append(subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(UI_APP), "--server.port", ui_port],
        cwd=settings.root, env=env,
    ))
    print(f"\nRisk Radar:  http://localhost:{ui_port}")
    if not args.no_api:
        print(f"Signal API:  http://localhost:{api_port}/docs")
    print("The default replay pack starts automatically. Ctrl-C to stop.\n")
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from seismo.engine import build_engine
    from seismo.schemas import Document, SourceType, stable_id

    engine = build_engine()
    now = datetime.now(UTC)
    doc = Document(
        doc_id=stable_id("d", "cli", args.text, now.isoformat()), source="cli",
        source_type=SourceType(args.source_type), publisher=args.publisher,
        published_at=now, title=args.title or "", body=args.text,
    )
    print(json.dumps([s.model_dump(mode="json") for s in engine.process(doc)], indent=2))
    return 0


def cmd_shocks(args: argparse.Namespace) -> int:
    from seismo.module_b.factors import main as shocks_main

    return shocks_main()


def cmd_blotter(args: argparse.Namespace) -> int:
    from seismo.module_b.blotter import main as blotter_main

    return blotter_main()


def cmd_stress(args: argparse.Namespace) -> int:
    from seismo.module_b.memo import write_memo
    from seismo.module_b.stress import StressEngine, StressRequest

    engine = StressEngine.from_settings(load_settings())
    req = StressRequest(args.scenario, args.impact, args.entity, args.event_class, args.subtype, args.m)
    result = engine.run(req, "cli")
    print(write_memo(result))
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    from seismo.schemas import Signal

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(Signal.model_json_schema(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seismo", description="Seismo AI/NLP risk engine")
    parser.add_argument("--version", action="version", version=f"seismo {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="Start the Risk Radar and the API with the default replay pack")
    p.add_argument("--no-api", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("replay", help="Replay a recorded JSONL pack through the pipeline")
    p.add_argument("--pack", help="Path to a .jsonl pack (default from config)")
    p.add_argument("--speed", type=float, help="Clock multiplier; 0 replays as fast as possible")
    p.add_argument("--reset", action="store_true", help="Clear stored signals first")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("live", help="Stream live sources (GDELT, EDGAR, Bluesky)")
    p.add_argument("--sources", default="gdelt,edgar,bluesky")
    p.add_argument("--minutes", type=float, help="Stop after N minutes")
    p.add_argument("--record", help="Also append every document to this .jsonl pack")
    p.add_argument("--reset", action="store_true")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("api", help="Serve the signal API (FastAPI)")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_api)

    p = sub.add_parser("ui", help="Open the Risk Radar dashboard")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("analyze", help="Score one piece of text and print the signals")
    p.add_argument("text")
    p.add_argument("--title", default="")
    p.add_argument("--publisher", default="user-input")
    p.add_argument("--source-type", default="news", choices=["news", "social", "filing"])
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("shocks", help="Compute analog shock vectors from data/market (Module B)")
    p.set_defaults(func=cmd_shocks)

    p = sub.add_parser("blotter", help="Write the synthetic trade blotter to data/blotter (Module B)")
    p.set_defaults(func=cmd_blotter)

    p = sub.add_parser("stress", help="Run one Module B stress test and print the memo")
    p.add_argument("--scenario", default="svb_2023")
    p.add_argument("--impact", type=int, default=9)
    p.add_argument("--entity", default=None, help="obligor at the centre, e.g. SIVB")
    p.add_argument("--event-class", default="CREDIT_EVENT")
    p.add_argument("--subtype", default="BANK_RUN")
    p.add_argument("--m", type=float, default=1.0, help="extra severity multiplier")
    p.set_defaults(func=cmd_stress)

    p = sub.add_parser("schema", help="Export the Signal JSON Schema")
    p.add_argument("--out", default="docs/signal_schema.json")
    p.set_defaults(func=cmd_schema)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return int(args.func(args) or 0)
