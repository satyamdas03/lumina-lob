"""Tests for the PPO vs SAC comparison helper."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from lumina_lob.rl.compare import compare_ppo_sac


def fake_env():
    return MagicMock()


def test_compare_ppo_sac_returns_both_results():
    with (
        patch("lumina_lob.rl.compare.train_ppo") as mock_ppo,
        patch("lumina_lob.rl.compare.train_sac") as mock_sac,
        patch("lumina_lob.rl.compare.evaluate_agent") as mock_eval,
    ):
        mock_ppo.return_value = "ppo_model"
        mock_sac.return_value = "sac_model"
        mock_eval.side_effect = [(1.0, 0.1), (2.0, 0.2)]

        result = compare_ppo_sac(fake_env, total_timesteps=64, n_eval_episodes=2)

        assert set(result.keys()) == {"ppo", "sac"}
        assert result["ppo"]["mean"] == 1.0
        assert result["ppo"]["std"] == 0.1
        assert result["sac"]["mean"] == 2.0
        assert result["sac"]["std"] == 0.2

        assert mock_ppo.call_count == 1
        assert mock_sac.call_count == 1
        assert mock_eval.call_count == 2
