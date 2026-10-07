import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "verl" / "verl" / "trainer" / "mtp" / "launch.py"
SPEC = importlib.util.spec_from_file_location("growmtp_mtp_launch_for_test", LAUNCH)
LAUNCH_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCH_MODULE)
initialize_mtp_head = getattr(LAUNCH_MODULE, "initialize_mtp_head", None)


def test_mtp_head_initialization_repeats_for_same_seed():
    if initialize_mtp_head is None:
        pytest.fail("MTP head initialization must accept the experiment seed")
    config = type("Config", (), {"initializer_range": 0.02})()
    first = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    second = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))

    initialize_mtp_head(first, config, seed=29)
    initialize_mtp_head(second, config, seed=29)

    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


def test_mtp_head_initialization_changes_for_a_different_seed():
    if initialize_mtp_head is None:
        pytest.fail("MTP head initialization must accept the experiment seed")
    config = type("Config", (), {"initializer_range": 0.02})()
    first = torch.nn.Sequential(torch.nn.Linear(4, 3))
    second = torch.nn.Sequential(torch.nn.Linear(4, 3))

    initialize_mtp_head(first, config, seed=29)
    initialize_mtp_head(second, config, seed=30)

    assert not torch.equal(first[0].weight, second[0].weight)
