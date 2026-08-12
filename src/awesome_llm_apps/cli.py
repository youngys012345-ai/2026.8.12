from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import ConfigError, load_config
from .orchestrator import build_plan, execute


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "plan", "tools", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("-c", "--config", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_config(args.config)
        if args.command == "validate":
            print("configuration is valid")
            return 0
        if args.command == "plan":
            print(json.dumps([asdict(step) for step in build_plan(config)], ensure_ascii=False, indent=2))
            return 0

        # Keep offline validation/planning usable before optional dependencies are installed.
        from .mcp import McpClient, McpError

        config.validate(require_environment=True)
        url, token = config.remote.resolve()
        with McpClient(
            url,
            token,
            timeout=config.remote.timeout_seconds,
            verify_tls=config.remote.verify_tls,
        ) as client:
            result = client.list_tools() if args.command == "tools" else execute(config, client)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
