"""Command line: python -m futsal {run,analyze,render,export} ..."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import config, pipeline, report
from .render import render

LOG = logging.getLogger("futsal")


def parse(argv=None):
    p = argparse.ArgumentParser(prog="futsal", description="Detect, track and summarize a futsal video.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Detect (slow) + analyze + render into a new run directory")
    run.add_argument("config", type=Path)
    run.add_argument("--video", type=Path, help="Override the video in the config")
    run.add_argument("--no-render", action="store_true")

    ana = sub.add_parser("analyze", help="Re-analyze an existing run with a (changed) config; no YOLO")
    ana.add_argument("config", type=Path)
    ana.add_argument("run_dir", type=Path)
    ana.add_argument("--render", action="store_true")

    ren = sub.add_parser("render", help="Draw an analyzed run over its video")
    ren.add_argument("run_dir", type=Path)
    ren.add_argument("--output", type=Path)

    exp = sub.add_parser("export", help="Write the coach-report data (heatmaps, timeline...) of an analyzed run")
    exp.add_argument("config", type=Path)
    exp.add_argument("run_dir", type=Path)
    exp.add_argument("--output", type=Path, help="Default: <run_dir>/report")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if a.command == "run":
            cfg = config.load(a.config)
            if a.video:
                cfg.video = a.video
            run_dir = pipeline.new_run_dir(cfg)
            LOG.info("Run directory: %s", run_dir)
            pipeline.detect(cfg, run_dir)
            pipeline.analyze(cfg, run_dir)
            if not a.no_render:
                render(run_dir)
        elif a.command == "analyze":
            pipeline.analyze(config.load(a.config), a.run_dir)
            if a.render:
                render(a.run_dir)
        elif a.command == "export":
            sizes = report.export(config.load(a.config), a.run_dir, a.output or a.run_dir / "report")
            LOG.info("Report data: %s", sizes)
        else:
            render(a.run_dir, a.output)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        LOG.error("%s", e)
        return 2
    except KeyboardInterrupt:
        LOG.warning("Interrupted; output may be incomplete.")
        return 130
    return 0
