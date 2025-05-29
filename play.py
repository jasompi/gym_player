#! /usr/bin/env python3

from agent import Agent, Experience
import argparse
import collections
import datetime
import gymnasium as gym
import gym_pygame
import importlib
import imageio
import logging
import numpy as np
import os
import sys
import time
import torch
from typing import List, Tuple, Optional

MAX_EXPERIENCES_SIZE = 100_000

def play_episode(env,
                 episode: int,
                 agent: Agent,
                 experiences: Optional[collections.deque[Experience]] = None,
                 frame_buffer: Optional[List[np.ndarray]] = None,
                 delay: int = 0) -> Tuple[float, int]:
    """Play a single episode in the environment using the given agent.
    Args:
        env (gym.Env): The environment to play in.
        episode (int): The episode number.
        agent (Agent): The agent to use for playing the episode.
        experiences (List[Experience]): List of experiences to collect.
        frame_buffer (List[np.ndarray]): Buffer to store frames for rendering.
    Returns:
        Tuple[float, int]: The total reward and the number of steps.
    """
    agent.train(experiences is not None)

    state, _ = env.reset()
    state = torch.from_numpy(state).float()
    action = agent.act(state)
    logging.debug(f'action: {action.action}; log_prob: {action.log_prob}; value: {action.value}')
    if frame_buffer is not None:
        frame_buffer.append(env.render())

    total_reward = 0
    done = False
    for i in range(env.spec.max_episode_steps):
        # this is where you would insert your policy

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        next_state, reward, done, truncated, _ = env.step(action.action.item())
        next_state = torch.from_numpy(next_state).float()
        next_action = agent.act(next_state)
        if frame_buffer is not None:
            frame_buffer.append(env.render())
        total_reward += reward
        if experiences is not None:
            next_value = next_action.value.clone().detach() if not done and next_action.value is not None else None
            experiences.append(Experience(state, action.action, reward, done, truncated, next_state, next_action.action, action.log_prob, action.value, next_value))
        logging.debug(f'{env.spec.id} Episode: {episode}:{i}; action: {action.action}; reward: {reward}; total_reward: {total_reward}; state: {state}; done: {done}; truncated: {truncated}, log_prob: {action.log_prob}; value: {action.value}')
        state = next_state
        action = next_action

        # If the episode has ended then we can reset to start a new episode
        if done or truncated:
            break
        if delay:
            time.sleep(delay / 1000)

    logging.debug(f'{env.spec.id} Episode: {episode}; terminated: {done} in {i} steps; total reward: {total_reward}')
    return total_reward, i + 1

def eval_output(e: int, episode_scores: List[float], agent_metrics: str, verbose: int=0):
    if episode_scores:
        mean_reward = np.mean(episode_scores)
        std_reward = np.std(episode_scores)
        if verbose > 0:
            print(f'Episode {e}: {len(episode_scores)} episodes rewards mean: {mean_reward:.2f}; std: {std_reward:.2f}; {agent_metrics}')
        return mean_reward, std_reward
    else:
        return 0, 0

def run_agent(env, n_episode: int, evaluation_episode_freq: int, agent, training_frequency: int, score: float, out_directory: Optional[str]=None, verbose=0, delay: int=0):
    verbose = max(verbose, 0 if training_frequency else 1) # if training_frequency is 0 (eval), verbose is at least 1
    scores_deque = collections.deque(maxlen=evaluation_episode_freq)
    experiences: collections.deque[Experience] = collections.deque(maxlen=MAX_EXPERIENCES_SIZE)
    frame_buffer: List[np.ndarray] = []
    total_steps = 0
    steps_since_last_train = 0
    start = time.time()
    try:
        for e in range(1, n_episode + 1):
            reward, steps = play_episode(env, e, agent, experiences if training_frequency else None, frame_buffer if out_directory and e % evaluation_episode_freq == 0 else None, delay)
            scores_deque.append(reward)
            total_steps += steps
            if training_frequency: # Accumulate steps if in training mode
                steps_since_last_train += steps

            if len(frame_buffer) and out_directory:
                filename = os.path.join(out_directory, f'{env.spec.id}-episode-{e}.mp4')
                imageio.mimwrite(filename, [np.array(img_frame) for _, img_frame in enumerate(frame_buffer)], fps=env.metadata['render_fps'])
                print(f"Video saved to {filename}")
                frame_buffer.clear()
            if training_frequency and experiences and e % training_frequency == 0:
                logging.info(f'{env.spec.id} Training after episode: {e}. Total experiences in buffer: {len(experiences)}. Cumulative untrained steps: {steps_since_last_train}.')
                agent.reinforce(experiences, steps_since_last_train)
                steps_since_last_train = 0 # Reset accumulator after training

            if e % evaluation_episode_freq == 0:
                mean_score, std_score = eval_output(e, list(scores_deque), agent.learning_metrics(), verbose=verbose)
                if mean_score - std_score >= score:
                    print(f"Stopping training at episode {e} with mean score: {mean_score:.2f}, std: {std_score:.2f}")
                    break
                scores_deque.clear()

        if scores_deque: # Check if scores_deque is not empty before eval_output
            eval_output(e, list(scores_deque), agent.learning_metrics(), verbose=verbose)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    total_time = time.time() - start
    if verbose > 0:
        print(f"Total time: {total_time:.2f} seconds for {total_steps} steps in {e} episodes. {total_steps / total_time:.2f} steps/sec")

def main(argv: List[str]):
    parser = argparse.ArgumentParser(description='Train and play gymnasium environment.')
    parser.add_argument('-v', '--verbose', action="count", default=0, help="Increase verbosity level")
    parser.add_argument('-e', '--envId', help='environment Id')
    parser.add_argument('-a', '--agent', default='RandomPolicy', help='agent module name')
    parser.add_argument('-f', '--filename', type=str, help='filename load the previous trained agent')
    parser.add_argument('-n', '--num-episode', type=int, help='number of episode')
    parser.add_argument('-m', '--max-step', type=int, help='max number of step in an episode')
    parser.add_argument('-u', '--evaluation-episode', type=int, help='show evaluation score every number of episode')
    parser.add_argument(
        '-t', '--train-freq',
        type=int,
        nargs='?',
        const=1,  # Value if flag is present without arg
        default=0, # Value if flag is not present
        metavar='N_EPISODES',
        help='Frequency of training, in episodes. '
             '0: Evaluation mode (no training). '
             '1 (or -t alone): Train after every episode. '
             'N (>1): Train after every N episodes. Default: 0.')
    parser.add_argument('-s', '--score', type=float, default=float('inf'), help='score threshold to stop training')
    parser.add_argument('-r', '--render', help='render mode. `display` to rander on display or PATH to record video for every episode and save it to PATH')
    parser.add_argument('-d', '--delay', type=int, default=0, help='delay microseconds between frames')
    parser.add_argument('--log', default=logging.WARNING, help='log level')

    args, unrecognized = parser.parse_known_args(argv)

    logging.basicConfig(level=args.log, format='%(asctime)s %(levelname)s %(message)s')
    
    render_mode = None
    out_directory = None
    if args.render == 'display':
        render_mode = 'human'
    elif args.render:
        render_mode = 'rgb_array'
        out_directory = args.render
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)

    state = {}
    if args.filename:
        state = torch.load(args.filename, weights_only=False)
    
    envId = state.get('env', args.envId)
    if not envId:
        print("Please provide an environment ID or a filename to load a environment.", file=sys.stderr)
        parser.print_help()
        exit(0)
    
    max_step = args.max_step or state.get('max_step', None) or 1000

    # Initialise the environment
    env = gym.make(envId, max_episode_steps=max_step, render_mode=render_mode)
    print(f'Initializing environment: {envId}')

    agent_module = state.get('agent', args.agent)
    try:
        pm = importlib.import_module(agent_module)
    except ImportError:
        print(f"Policy module {agent_module} not found.", file=sys.stderr)
        exit(1)

    agent = pm.load_agent(env, state) if state else pm.create_agent(env, unrecognized)

    training_frequency = args.train_freq
    if training_frequency < 0:
        parser.error("--train-freq cannot be negative.")

    max_step_env = env.spec.max_episode_steps or max_step # type: ignore

    if training_frequency: # Non-zero training_frequency means training mode
        actual_n_episode = args.num_episode or 1000  # Default training episodes
        print(f'train {agent_module} for {envId} for {actual_n_episode} episodes with {max_step_env} steps, training every {training_frequency} episode(s).')
        eval_episode_freq = args.evaluation_episode or int(actual_n_episode / 25) or actual_n_episode
    else:  # training_frequency == 0, evaluation mode
        actual_n_episode = args.num_episode or 10  # Default evaluation episodes
        print(f'eval {agent_module} for {envId} for {actual_n_episode} episodes with {max_step_env} steps.')
        eval_episode_freq = actual_n_episode # Evaluate once at the end for eval mode

    run_agent(env, actual_n_episode, eval_episode_freq, agent, training_frequency, score=args.score, verbose=args.verbose, out_directory=out_directory, delay=args.delay)
    if training_frequency: # Non-zero training_frequency means training mode
        filename = args.filename
        if state or not filename:
            filename = f'{envId}-{agent_module}-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pt'
        state = agent.get_state_dict()
        state['env'] = envId
        state['agent'] = type(agent).__module__
        state['max_step'] = max_step
        torch.save(state, filename)
        print(f"Agent saved to {filename}")

    env.close()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])