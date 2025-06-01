# Gymnasium Player

Experiment agents with reinforcement learning algorithms:

* Deep Q-Network
* Policy Gradient
* Actor Critic

* Monte Carlo vs TD(n)

Gym Environment:

* CartPole-v1

 https://github.com/user-attachments/assets/5be94564-3154-4854-8c5c-2e4a73d7dcaf

* LunarLander-v3

 https://github.com/user-attachments/assets/ed647616-7342-4264-9a6b-5bf4b49da9ab

* Pixelcopter

 https://github.com/user-attachments/assets/7ed501fb-5aa0-422d-bc89-d32ce0a2de90

* Pong

https://github.com/user-attachments/assets/ea8dced0-e82e-4267-b339-76a6ddfc0863
  
## Setup
```
sudo apt install swig
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train an agent:
```
$ ./play.py -e CartPole -a DeepQNetwork -t -v -n 10000 -s 400
Creating DeepQNetwork for CartPole-v1 with layers: [16], gamma: 1.0, lr: 0.01
train DeepQNetwork for CartPole for 10000 episodes with 500 steps
Episode 400: 400 episodes rewards mean: 21.57; std: 10.31
Episode 800: 400 episodes rewards mean: 26.19; std: 15.94
Episode 1200: 400 episodes rewards mean: 38.67; std: 26.83
Episode 1600: 400 episodes rewards mean: 86.64; std: 65.15
Episode 2000: 400 episodes rewards mean: 196.78; std: 92.16
Episode 2400: 400 episodes rewards mean: 240.79; std: 182.08
Episode 2800: 400 episodes rewards mean: 189.97; std: 150.91
Episode 3200: 400 episodes rewards mean: 408.94; std: 167.36
Stopping training at episode 3200 with mean score: 408.94, std: 167.36
Episode 3200: 400 episodes rewards mean: 408.94; std: 167.36
Agent saved to CartPole-DeepQNetwork-20250511135443.pt
```

## Eval an agent
```
$ ./play.py -f CartPole-DeepQNetwork-20250511135443.pt -n 100
Loading DeepQNetwork for CartPole-v1 with layers: [16], gamma: 1.0, lr: 0.01, tau: 0.001, epsilon_decay: 0.95, epsilon: 0.05, total_updates: 119931
eval DeepQNetwork for CartPole for 100 episodes with 500 steps
Episode 100: 100 episodes rewards mean: 500.00; std: 0.00
```

## Arguments

```
$ ./play.py -h
usage: play.py [-h] [-v] [-e ENVID] [-a AGENT] [-f FILENAME] [-n NUM_EPISODE] [-m MAX_STEP] [-u EVALUATION_EPISODE] [-t] [-s SCORE] [-r RENDER] [--log LOG]

Train and play gymnasium environment.

options:
  -h, --help            show this help message and exit
  -v, --verbose         Increase verbosity level
  -e ENVID, --envId ENVID
                        environment Id
  -a AGENT, --agent AGENT
                        agent module name
  -f FILENAME, --filename FILENAME
                        filename load the previous trained agent
  -n NUM_EPISODE, --num-episode NUM_EPISODE
                        number of episode
  -m MAX_STEP, --max-step MAX_STEP
                        max number of step in an episode
  -u EVALUATION_EPISODE, --evaluation-episode EVALUATION_EPISODE
                        show evaluation score every number of episode
  -t, --train           train the agent
  -s SCORE, --score SCORE
                        score threshold to stop training
  -r RENDER, --render RENDER
                        render mode. `display` to rander on display or PATH to record video for every episode and save it to PATH
  --log LOG             log level
```
