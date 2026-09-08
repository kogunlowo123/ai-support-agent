"""The command line, driven the way CI and an operator drive it.

The exit code is the contract that matters. ``evaluate`` returning zero on a
failing suite is the difference between a gate and a report nobody reads, so
that is asserted directly rather than inferred from the printed output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from support_agent import cli
from support_agent.config import (
    ChatBackend,
    ChatSettings,
    Environment,
    SecuritySettings,
    Settings,
    StorageSettings,
)

pytestmark = pytest.mark.integration

SUITE = Path(__file__).resolve().parents[2] / "data" / "scenarios" / "support.jsonl"


@pytest.fixture
def cli_settings(tmp_path: Path, monkeypatch) -> Settings:
    """Point the CLI at a database of its own, and stop it reading the environment."""
    settings = Settings(
        environment=Environment.TEST,
        storage=StorageSettings(
            database_url=f"sqlite+aiosqlite:///{(tmp_path / 'cli.db').as_posix()}"
        ),
        chat=ChatSettings(backend=ChatBackend.TEMPLATE),
        security=SecuritySettings(require_api_key=False),
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return settings


class TestArgumentParsing:
    def test_the_parser_builds(self):
        parser = cli.build_parser()
        assert parser.prog == "support-agent"

    def test_every_command_is_registered(self):
        parser = cli.build_parser()
        subcommands = next(action for action in parser._actions if getattr(action, "choices", None))
        commands = set(subcommands.choices or {})
        assert {"serve", "seed", "chat", "evaluate"} <= commands

    def test_no_command_prints_help_and_exits_non_zero(self, capsys):
        assert cli.main([]) == 2
        assert "usage" in capsys.readouterr().out.lower()

    def test_an_unknown_command_is_rejected(self):
        with pytest.raises(SystemExit):
            cli.main(["nonsense"])

    def test_the_version_is_reported(self, capsys):
        with pytest.raises(SystemExit) as raised:
            cli.main(["--version"])
        assert raised.value.code == 0
        assert "support-agent" in capsys.readouterr().out


class TestSeed:
    def test_it_loads_the_knowledge_base_and_the_demo_account(self, cli_settings, capsys):
        assert cli.main(["seed"]) == 0
        output = capsys.readouterr().out
        assert "articles added   6" in output
        assert "ORD-1001" in output

    def test_running_it_twice_adds_nothing_the_second_time(self, cli_settings, capsys):
        cli.main(["seed"])
        capsys.readouterr()
        cli.main(["seed"])
        assert "articles added   0" in capsys.readouterr().out

    def test_a_tenant_can_be_named(self, cli_settings, capsys):
        assert cli.main(["--tenant", "globex", "seed"]) == 0
        assert "tenant           globex" in capsys.readouterr().out


class TestEvaluate:
    def test_a_passing_suite_exits_zero(self, cli_settings, capsys):
        code = cli.main(["evaluate", "--suite", str(SUITE), "--min-pass-rate", "1.0"])
        assert code == 0, capsys.readouterr().out

    def test_the_summary_is_printed(self, cli_settings, capsys):
        cli.main(["evaluate", "--suite", str(SUITE)])
        output = capsys.readouterr().out
        assert "pass rate" in output
        assert "adversarial" in output

    def test_an_impossible_threshold_fails_the_gate(self, cli_settings, capsys):
        """The property that makes this a gate: it can actually fail."""
        code = cli.main(["evaluate", "--suite", str(SUITE), "--max-escalation-rate", "0.0"])
        assert code == 1
        assert "gate failures" in capsys.readouterr().out

    def test_a_report_is_written_for_ci(self, cli_settings, tmp_path, capsys):
        report = tmp_path / "reports" / "scenarios.json"
        cli.main(["evaluate", "--suite", str(SUITE), "--report", str(report)])
        capsys.readouterr()

        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["total"] >= 20
        assert payload["green"] is True
        assert {"suite", "pass_rate", "categories", "scenarios"} <= set(payload)

    def test_the_verbose_flag_shows_passing_scenarios(self, cli_settings, capsys):
        cli.main(["evaluate", "--suite", str(SUITE), "--verbose"])
        assert "PASS" in capsys.readouterr().out

    def test_a_missing_suite_file_is_reported(self, cli_settings, tmp_path):
        with pytest.raises(FileNotFoundError):
            cli.main(["evaluate", "--suite", str(tmp_path / "nope.jsonl")])

    def test_an_empty_suite_file_is_rejected(self, cli_settings, tmp_path):
        """A suite that asserts nothing must not report itself as green."""
        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            cli.main(["evaluate", "--suite", str(empty)])
