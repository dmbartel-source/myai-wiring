"""Tests for tools/local_gen.py — the self-hosted diffusion stub.

The stub must refuse loudly until explicitly enabled AND healthy.
No network beyond loopback health probes (which fail fast here).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import local_gen
from local_gen import LocalGenConfig, LocalGenNotWired, enabled, health_check
from net import EgressBlocked


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("MYAI_LOCAL_GEN", raising=False)
    monkeypatch.delenv("MYAI_LOCAL_GEN_URL", raising=False)


def test_disabled_by_default():
    assert enabled() is False
    with pytest.raises(LocalGenNotWired, match="not enabled"):
        local_gen.generate("a cat astronaut")


def test_enabled_but_no_server_refuses():
    os.environ["MYAI_LOCAL_GEN"] = "1"
    # nothing listens on 127.0.0.1:8188 here -> health check fails fast
    assert health_check(LocalGenConfig(url="http://127.0.0.1:9")) is False
    with pytest.raises(LocalGenNotWired, match="health check"):
        local_gen.generate("a cat astronaut",
                           LocalGenConfig(url="http://127.0.0.1:9"))


def test_non_loopback_url_refused_by_egress_gate():
    cfg = LocalGenConfig(url="https://gen.example.com")
    with pytest.raises(EgressBlocked):
        cfg.validate()


def test_unknown_kind_refused():
    cfg = LocalGenConfig(url="http://127.0.0.1:8188", kind="telepathy")
    with pytest.raises(LocalGenNotWired, match="unknown local gen kind"):
        cfg.validate()


def test_config_from_env():
    os.environ["MYAI_LOCAL_GEN_URL"] = "http://127.0.0.1:8188"
    cfg = LocalGenConfig.from_env()
    assert cfg.url == "http://127.0.0.1:8188"
    cfg.validate()  # loopback: passes validation
