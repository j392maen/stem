from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from stemapp import cli, doctor
from stemapp.config import Settings
from stemapp.db import make_engine, make_session_factory
from stemapp.doctor import CheckResult, Status, exit_code, run_checks
from stemapp.seed import seed


def _fixed(name: str, status: Status) -> doctor.Check:
    def check(_s: Settings) -> CheckResult:
        return CheckResult(name, status, "detail", "hint")

    check.__name__ = name
    return check


def _boom(_s: Settings) -> CheckResult:
    raise RuntimeError("壊れた")


def test_exit_code_ok_and_warn(settings: Settings) -> None:
    results = run_checks(settings, [_fixed("a", Status.OK), _fixed("b", Status.WARN)])
    assert [r.status for r in results] == [Status.OK, Status.WARN]
    assert exit_code(results) == 0


def test_exit_code_ng(settings: Settings) -> None:
    results = run_checks(settings, [_fixed("a", Status.OK), _fixed("b", Status.NG)])
    assert exit_code(results) == 1


def test_failing_check_becomes_ng(settings: Settings) -> None:
    results = run_checks(settings, [_boom])
    assert results[0].status is Status.NG
    assert "壊れた" in results[0].detail
    assert exit_code(results) == 1


def test_required_checks_ok(settings: Settings) -> None:
    assert doctor.check_python(settings).status is Status.OK
    assert doctor.check_data_dir(settings).status is Status.OK
    # 初期データが無い DB は「注意」、init-db 後は OK
    assert doctor.check_db(settings).status is Status.WARN
    engine = make_engine(settings.db_path)
    with make_session_factory(engine)() as session:
        seed(session)
    engine.dispose()
    assert doctor.check_db(settings).status is Status.OK


def test_data_dir_unwritable_is_ng(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    s = Settings(_env_file=None, data_dir=blocker / "sub")  # type: ignore[call-arg]
    assert doctor.check_data_dir(s).status is Status.NG
    assert doctor.check_db(s).status is Status.NG


def test_missing_tools_are_warn(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)
    assert doctor.check_ffmpeg(settings).status is Status.WARN
    assert doctor.check_deno(settings).status is Status.WARN

    s = Settings(_env_file=None, data_dir=tmp_path, ytdlp_path=tmp_path / "none.exe")  # type: ignore[call-arg]
    res = doctor.check_ytdlp(s)
    assert res.status is Status.WARN
    assert res.hint


def test_ytdlp_exe_ok(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "yt-dlp.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(doctor, "_run", lambda cmd, timeout=30.0: "2026.01.01")
    s = Settings(_env_file=None, data_dir=tmp_path, ytdlp_path=exe)  # type: ignore[call-arg]
    res = doctor.check_ytdlp(s)
    assert res.status is Status.OK
    assert "2026.01.01" in res.detail


def _invoke_doctor(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, checks: list[doctor.Check]
) -> tuple[int, str]:
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_checks", lambda s: run_checks(s, checks))
    result = CliRunner().invoke(cli.app, ["doctor"])
    return result.exit_code, result.output


def test_cli_doctor_exit_codes(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _invoke_doctor(
        monkeypatch, settings, [_fixed("aaa", Status.OK), _fixed("bbb", Status.WARN)]
    )
    assert code == 0
    assert "aaa" in out and "注意" in out

    code, out = _invoke_doctor(monkeypatch, settings, [_fixed("ccc", Status.NG)])
    assert code == 1
    assert "NG" in out


def test_cli_init_db(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["init-db"]).exit_code == 0
    assert runner.invoke(cli.app, ["init-db"]).exit_code == 0
    assert settings.db_path.is_file()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_host_local_is_ok(tmp_path: Path, host: str) -> None:
    s = Settings(_env_file=None, data_dir=tmp_path / "d", host=host)  # type: ignore[call-arg]
    res = doctor.check_host(s)
    assert res.status is Status.OK and res.name == "待ち受けアドレス"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
def test_host_public_is_warned(tmp_path: Path, host: str) -> None:
    s = Settings(_env_file=None, data_dir=tmp_path / "d", host=host)  # type: ignore[call-arg]
    res = doctor.check_host(s)
    assert res.status is Status.WARN
    assert "外部に公開" in res.detail and "Tailscale Serve" in res.hint


def test_host_check_is_in_default_checks() -> None:
    assert doctor.check_host in doctor.DEFAULT_CHECKS


# --- Tailscale ------------------------------------------------------------------------

TS = "unagi.tail8b25a2.ts.net"
TS_STATUS = {"BackendState": "Running", "Self": {"DNSName": f"{TS}."}}


def _serve(port: int = 8000, funnel: bool = False) -> dict[str, object]:
    conf: dict[str, object] = {
        "TCP": {"443": {"HTTPS": True}},
        "Web": {f"{TS}:443": {"Handlers": {"/": {"Proxy": f"http://127.0.0.1:{port}"}}}},
    }
    if funnel:
        conf["AllowFunnel"] = {f"{TS}:443": True}
    return conf


def _fake_tailscale(
    monkeypatch: pytest.MonkeyPatch, status: object, serve: object = None
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake(_exe: str, args: list[str]) -> object:
        calls.append(list(args))
        if args[0] == "status":
            if isinstance(status, Exception):
                raise status
            return status
        if isinstance(serve, Exception):
            raise serve
        return {} if serve is None else serve

    monkeypatch.setattr(doctor, "find_tailscale", lambda: "tailscale.exe")
    monkeypatch.setattr(doctor, "_tailscale_json", fake)
    return calls


def test_tailscale_missing_is_warn(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "find_tailscale", lambda: None)
    res = doctor.check_tailscale(settings)
    assert res.status is Status.WARN and "入っていません" in res.detail


def test_tailscale_not_logged_in(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tailscale(monkeypatch, {"BackendState": "NeedsLogin"})
    res = doctor.check_tailscale(settings)
    assert res.status is Status.WARN and "ログイン" in res.detail


def test_tailscale_status_error(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tailscale(monkeypatch, OSError("起動していない"))
    assert doctor.check_tailscale(settings).status is Status.WARN


def test_tailscale_running_without_serve(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tailscale(monkeypatch, TS_STATUS, {})
    res = doctor.check_tailscale(settings.model_copy(update={"allowed_hosts": ""}))
    assert res.status is Status.OK
    assert TS in res.detail and "serve 未設定" in res.detail
    assert f"STEMAPP_ALLOWED_HOSTS={TS}" in res.hint  # 名前を見つけて案内する
    assert "tailscale-serve.ps1 start" in res.hint


def test_tailscale_serve_on_and_allowed(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tailscale(monkeypatch, TS_STATUS, _serve())
    res = doctor.check_tailscale(settings.model_copy(update={"allowed_hosts": TS}))
    assert res.status is Status.OK
    assert f"https://{TS}:443/" in res.detail
    assert res.hint == ""


def test_tailscale_serve_on_but_host_not_allowed(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tailscale(monkeypatch, TS_STATUS, _serve())
    res = doctor.check_tailscale(settings.model_copy(update={"allowed_hosts": ""}))
    assert res.status is Status.WARN
    assert "STEMAPP_ALLOWED_HOSTS" in res.hint


def test_tailscale_serve_other_port_is_not_ours(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tailscale(monkeypatch, TS_STATUS, _serve(port=9999))
    res = doctor.check_tailscale(settings.model_copy(update={"allowed_hosts": TS}))
    assert "serve 未設定" in res.detail


def test_tailscale_funnel_is_warned(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tailscale(monkeypatch, TS_STATUS, _serve(funnel=True))
    res = doctor.check_tailscale(settings.model_copy(update={"allowed_hosts": TS}))
    assert res.status is Status.WARN
    assert "Funnel" in res.detail and "funnel reset" in res.hint


def test_tailscale_serve_status_error(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tailscale(monkeypatch, TS_STATUS, ValueError("壊れた JSON"))
    res = doctor.check_tailscale(settings)
    assert res.status is Status.OK
    assert "取れません" in res.detail


def test_tailscale_check_is_in_default_checks() -> None:
    assert doctor.check_tailscale in doctor.DEFAULT_CHECKS
