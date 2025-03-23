import pytest
import numpy as np

from src.PPO_memory import PPOMemory

class TestPPOMemory():

    def test_generate_batches(self):
        memory = PPOMemory(batch_size=5)
        for _ in range(20):
            memory.states.append(np.random.rand(4, 84, 84))
            memory.actions.append(np.random.randint(0, 2))
            memory.probs.append(np.random.rand())
            memory.vals.append(np.random.rand())
            memory.rewards.append(np.random.rand())
            memory.dones.append(np.random.choice([True, False]))

        states, actions, probs, vals, rewards, dones, batches = memory.generate_batches()

        assert states.shape == (20, 4, 84, 84)
        assert len(actions) == 20
        assert len(probs) == 20
        assert len(vals) == 20
        assert len(rewards) == 20
        assert len(dones) == 20
        assert len(batches) == 4
        for batch in batches:
            assert len(batch) == 5