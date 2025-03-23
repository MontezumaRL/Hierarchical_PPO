import pytest

from src.environnement import MontezumaEnvironment



class TestEnvironnement:

    def test_environnement(self):
        assert True
    
    def test_montezuma_environment(self):
        
        env = MontezumaEnvironment(render_mode='human')
        state = env.reset()
        assert state.shape == (4, 84, 84)  # Assuming preprocess_frame outputs 84x84 frames

        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        assert next_state.shape == (4, 84, 84)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        env.close()