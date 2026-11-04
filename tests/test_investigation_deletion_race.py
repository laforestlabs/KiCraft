from kicraft.server.accounts import AccountStore
from kicraft.server import investigate_runner


def test_deleted_investigation_cannot_recreate_log_or_launch(tmp_path, monkeypatch):
    store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
    uid = store.create_user("delete@example.com", "password-password").id
    report_id = store.create_support_report(user_id=uid, board_code="KC-ABC123")
    log_dir = tmp_path / "support_investigations"
    log_dir.mkdir()
    log = log_dir / f"investigate_report_{report_id}.log"
    inv_id = store.create_investigation(report_id=report_id, log_path=str(log))
    original_get = store.get_investigation
    first = True

    def delete_after_read(identifier):
        nonlocal first
        row = original_get(identifier)
        if first:
            first = False
            store.delete_user(uid)
        return row

    monkeypatch.setattr(store, "get_investigation", delete_after_read)
    monkeypatch.setattr(investigate_runner, "_omp_bin", lambda: "unused-omp")
    monkeypatch.setattr(investigate_runner.subprocess, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("launched")))
    rc, _ = investigate_runner._run_omp(store, inv_id, "KC-ABC123")
    assert rc is None
    assert not log.exists()
    assert original_get(inv_id) is None
