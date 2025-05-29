import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import unittest

import gymnasium as gym
import io
import numpy as np
import os
import play
import re
import sys
import tempfile
import torch
import PolicyGradient

def compute_returns_vec(rewards: torch.Tensor, done: bool, next_value: float, gamma: float = 1.0) -> torch.Tensor:
    n_step = len(rewards)
    if not done:
        rewards[-1] += next_value * gamma
    mask = torch.ones((n_step, n_step)).triu()
    power = torch.arange(n_step)
    t_returns = (torch.pow(gamma, mask * power - power.unsqueeze(1)) * mask * rewards).sum(dim=1)
    return t_returns

class TestPlayMain(unittest.TestCase):
    def setUp(self) -> None:
        self.cwd = os.getcwd()
        
    def tearDown(self) -> None:
        os.chdir(self.cwd)

    def run_main(self, argv) -> list[str]:
        # print(' '.join(argv))
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        play.main(argv)
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        print(output)
        lines = output.split('\n')
        for i in range(len(lines)):
            if lines[i].startswith('Initializing environment: '):
                break
        return lines[i+1:]

    def run_test(self, env_id: str, agent: str, tmpdir: str, max_steps: int=20):
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        os.chdir(tmpdir)
        video_dir =  'videos'
        lines = self.run_main([f'-e={env_id}', f'-a={agent}', '-n=3', f'-m={max_steps}', f'-r={video_dir}', '-u=1', '-t', '--log=ERROR'])

        self.assertTrue(os.path.exists(video_dir), "video directory not created")
        self.assertRegex(lines[0], fr'Creating {agent} for {env_id}')
        self.assertIn(f'train {agent} for {env_id} for 3 episodes with {max_steps} steps', lines[1])
        for i in range(1, 4):
            self.assertIn(f'{env_id}-episode-{i}.mp4', lines[i+1])
            self.assertTrue(os.path.exists(f'{video_dir}/{env_id}-episode-{i}.mp4'), f"video file {i} not created")
    
        pattern = fr'{env_id}-{agent}-\d+\.pt'
        match = re.search(pattern, lines[5])
        self.assertIsNotNone(match, "Filename pattern not found in output")
        filename = match.group(0) # type: ignore
        self.assertTrue(os.path.exists(filename), "policy file not created")
        
        lines = self.run_main([f'-f={filename}', '-t', '--log=ERROR'])
        self.assertRegex(lines[0], fr'Loading {agent} for {env_id}')
        self.assertIn(f'train {agent} for {env_id} for 1000 episodes with {max_steps} steps', lines[1])

        match = re.search(pattern, lines[2])
        self.assertIsNotNone(match, "Filename pattern not found in output")
        filename1 = match.group(0) # type: ignore
        self.assertTrue(os.path.exists(filename1), "policy file not created")
    
        lines = self.run_main([f'-f={filename1}', '-r=display', '-n=2', '-v', '--log=ERROR'])
        self.assertRegex(lines[0], fr'Loading {agent} for {env_id}')
        self.assertIn(f'eval {agent} for {env_id} for 2 episodes with {max_steps} steps', lines[1])
        self.assertRegex(lines[2], r'2 episodes rewards mean: (-?\d+\.\d+); std: (\d+\.\d+)')
        
    def test_CartPole_RandomPolicy(self):
        env_id = 'CartPole-v1'
        policy = 'RandomPolicy'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_CartPole_PolicyGradient(self):
        env_id = 'CartPole-v1'
        policy = 'PolicyGradient'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)
            
    def test_CartPole_ActorCritic(self):
        env_id = 'CartPole-v1'
        policy = 'ActorCriticMonteCarlo'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_CartPole_ActorCriticTD(self):
        env_id = 'CartPole-v1'
        policy = 'ActorCriticTD'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_CartPole_DeepQNetwork(self):
        env_id = 'CartPole-v1'
        policy = 'DeepQNetwork'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_CartPole_SARSA(self):
        env_id = 'CartPole-v1'
        policy = 'SARSA'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_LunaarLander_RandomPolicy(self):
        env_id = 'LunarLander-v3'
        policy = 'RandomPolicy'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_LunarLander_PolicyGradient(self):
        env_id = 'LunarLander-v3'
        policy = 'PolicyGradient'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_LunarLander_ActorCritic(self):
        env_id = 'LunarLander-v3'
        policy = 'ActorCriticMonteCarlo'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_LunarLander_ActorCriticTD(self):
        env_id = 'LunarLander-v3'
        policy = 'ActorCriticTD'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_LunarLander_DeepQNetwork(self):
        env_id = 'LunarLander-v3'
        policy = 'DeepQNetwork'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_LunarLander_SARSA(self):
        env_id = 'LunarLander-v3'
        policy = 'SARSA'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_Pixelcopter_RandomPolicy(self):
        env_id = 'Pixelcopter-PLE-v0'
        policy = 'RandomPolicy'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_Pixelcopter_PolicyGradient(self):
        env_id = 'Pixelcopter-PLE-v0'
        policy = 'PolicyGradient'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)
            
    def test_Pixelcopter_ActorCritic(self):
        env_id = 'Pixelcopter-PLE-v0'
        policy = 'ActorCriticMonteCarlo'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_Pixelcopter_ActorCriticTD(self):
        env_id = 'Pixelcopter-PLE-v0'
        policy = 'ActorCriticTD'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_Pixelcopter_DeepQNetwork(self):
        env_id = 'Pixelcopter-PLE-v0'
        policy = 'DeepQNetwork'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)

    def test_Pixelcopter_SARSA(self):
        env_id = 'Pixelcopter-PLE-v0'
        policy = 'SARSA'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_test(env_id, policy, tmpdir)
            
    def test_PolicyGradient_compute_returns(self):
        env_id = 'CartPole-v1'
        env = gym.make(env_id)
        _, _ = env.reset()
        n_steps = 10
        
        rewards = torch.ones((3, n_steps), dtype=torch.float32)
        dones = torch.zeros((3, n_steps), dtype=torch.long)
        truncates = torch.zeros((3, n_steps), dtype=torch.long)
        next_values = torch.ones((3, n_steps), dtype=torch.float32) * 0.5
        dones[0, -1] = 1
        truncates[1, -1] = 1
        
        agent: PolicyGradient.PolicyGradientAgent = PolicyGradient.create_agent(env, []) # type: ignore
        returns = agent.compute_returns(rewards, dones, truncates, next_values)
        
        expected_returns_0 = torch.arange(n_steps, 0, -1, dtype=torch.float32).unsqueeze(0)
        expected_returns_1 = expected_returns_0 + 0.5
        expected_returns = torch.cat((expected_returns_0, expected_returns_1, expected_returns_1), dim=0)
        self.assertTrue(torch.equal(returns, expected_returns))

        rewards2= torch.cat((rewards, rewards), dim=1)
        dones2= torch.cat((dones, dones), dim=1)
        truncates2= torch.cat((truncates, truncates), dim=1)
        next_values2= torch.cat((next_values, next_values), dim=1)
        returns2 = agent.compute_returns(rewards2, dones2, truncates2, next_values2)
        
        expected_returns_0_2 = torch.cat((expected_returns_0, expected_returns_0), dim=1)
        expected_returns_1_2 = torch.cat((expected_returns_1, expected_returns_1), dim=1)
        expected_returns_2_2 = torch.arange(n_steps * 2, 0, -1, dtype=torch.float32).unsqueeze(0) + 0.5
        expected_returns2 = torch.cat((expected_returns_0_2, expected_returns_1_2, expected_returns_2_2), dim=0)
        self.assertTrue(torch.equal(returns2, expected_returns2))
        
        agent._gamma = 0.9
        returns = agent.compute_returns(rewards, dones, truncates, next_values)
        expected_returns_0 = compute_returns_vec(rewards[0], True, 0.5, agent._gamma)
        expected_returns_1 = compute_returns_vec(rewards[1], False, 0.5, agent._gamma)
        expected_returns = torch.stack((expected_returns_0, expected_returns_1, expected_returns_1))
        self.assertTrue(torch.allclose(returns, expected_returns))

        returns2 = agent.compute_returns(rewards2, dones2, truncates2, next_values2)
        expected_returns_0_2 = torch.cat((expected_returns_0, expected_returns_0), dim=0)
        expected_returns_1_2 = torch.cat((expected_returns_1, expected_returns_1), dim=0)
        expected_returns_2_2 = compute_returns_vec(rewards2[2], False, 0.5, agent._gamma)
        expected_returns2 = torch.stack((expected_returns_0_2, expected_returns_1_2, expected_returns_2_2), dim=0)
        self.assertTrue(torch.allclose(returns2, expected_returns2))
        
    def test_PolicyGradient_mean_variance(self):
        n_samples = np.random.randint(1, 10, 5).tolist()
        samples = [np.random.random_sample(n) for n in n_samples]
        means = [s.mean() for s in samples]
        variances = [s.var() for s in samples]
        mean, variance, n_sample = PolicyGradient.mean_variance(means, variances, n_samples)
        sample = np.concatenate(samples)
        self.assertAlmostEqual(mean, sample.mean())
        self.assertAlmostEqual(variance, sample.var())
        self.assertEqual(n_sample, np.sum(n_samples))
        
        mean1, variance1, n_sample1 = PolicyGradient.mean_variance([mean, 0], [variance, 0], [n_sample, 0])
        self.assertAlmostEqual(mean, mean1)
        self.assertAlmostEqual(variance, variance1)
        self.assertEqual(n_sample, n_sample1)
        
if __name__ == '__main__':
    unittest.main()