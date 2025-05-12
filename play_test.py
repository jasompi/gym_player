import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import unittest

import io
import os
import play
import re
import sys
import tempfile

class TestPlayMain(unittest.TestCase):
    def run_main(self, argv) -> str:
        # print(' '.join(argv))
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        play.main(argv)
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        # print(output)
        return output

    def run_test(self, env_id: str, agent: str, tmpdir: str, max_steps: int=20):
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        os.chdir(tmpdir)
        video_dir =  'videos'
        output = self.run_main([f'-e={env_id}', f'-a={agent}', '-n=3', f'-m={max_steps}', f'-r={video_dir}', '-t', '--log=ERROR'])

        self.assertTrue(os.path.exists(video_dir), "video directory not created")
        lines = output.split('\n')
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
        
        output = self.run_main([f'-f={filename}', '-t', '-l=ERROR'])
        lines = output.split('\n')
        self.assertRegex(lines[0], fr'Loading {agent} for {env_id}')
        self.assertIn(f'train {agent} for {env_id} for 1000 episodes with {max_steps} steps', lines[1])

        match = re.search(pattern, lines[2])
        self.assertIsNotNone(match, "Filename pattern not found in output")
        filename1 = match.group(0) # type: ignore
        self.assertTrue(os.path.exists(filename1), "policy file not created")
    
        output = self.run_main([f'-f={filename1}', '-r=display', '-n=2', '-v', '-l=ERROR'])
        lines = output.split('\n')
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

if __name__ == '__main__':
    unittest.main()