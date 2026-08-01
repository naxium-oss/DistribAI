from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class BenchmarkManager:
    """
    Wraps the existing bench_runner.py to execute the full DistribAI
    benchmark suite and return the consolidated report.
    """

    def __init__(self, node_id: str):
        self.bench_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark"))
        self.runner_path = os.path.join(self.bench_dir, "bench_runner.py")
        self.node_id = node_id

    async def run_full_suite(self) -> dict | None:
        """
        Runs the benchmark runner as a subprocess and parses the final suite_complete message.
        """
        logger.info("Starting DistribAI Benchmark Suite (this may take ~60s)...")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            self.runner_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        final_report = None
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    msg_type = data.get("type")
                    if msg_type == "suite_progress":
                        logger.info(
                            f"Benchmark progress: {data.get('current')}/{data.get('total')} - {data.get('current_name')}"
                        )
                    elif msg_type == "suite_complete":
                        final_report = data
                        results_file = (
                            Path(tempfile.gettempdir())
                            / f"distribai_benchmark_results_{self.node_id}.json"
                        )
                        results_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                        logger.info(
                            f"Benchmark Complete! Overall Score: {data.get('overall_score')}"
                        )
                    elif msg_type == "error":
                        logger.error(
                            f"Benchmark Error in {data.get('name')}: {data.get('message')}"
                        )
                except json.JSONDecodeError:
                    pass
            await process.wait()
            if process.returncode != 0:
                stderr = await process.stderr.read()
                logger.error(
                    f"Benchmark Runner failed with code {process.returncode}: {stderr.decode()}"
                )
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
                await process.wait()
            raise
        except Exception as e:
            logger.error(f"Error running benchmark: {e}", exc_info=True)
            if process.returncode is None:
                process.terminate()
                await process.wait()
        finally:
            if process.stdout:
                process.stdout.feed_eof()
            if process.stderr:
                process.stderr.feed_eof()
        return final_report
