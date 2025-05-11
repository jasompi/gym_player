#! /usr/bin/env python3

from agent import Agent, Experience
import argparse
import collections
import datetime
import gymnasium as gym
import importlib
import imageio
import logging
import numpy as np
import os
import sys
import torch
from typing import List, Dict, Tuple, Any, Optional

MAX_EXPERIENCES_SIZE = 10000

def play_episode(env, episode: int, agent: Agent, experiences: Optional[collections.deque[Experience]] = None, frame_buffer: Optional[List[np.ndarray]] = None) -> float:
    """Play a single episode in the environment using the given agent.
    Args:
        env (gym.Env): The environment to play in.
        episode (int): The episode number.
        agent (Agent): The agent to use for playing the episode.
        experiences (List[Experience]): List of experiences to collect.
        frame_buffer (List[np.ndarray]): Buffer to store frames for rendering.
    Returns:
        Tuple[float, List[np.ndarray]]: The total reward and a list of frames if rendered.
    """
    agent.train(experiences is not None)

    state, _ = env.reset()
    if frame_buffer is not None:
        frame_buffer.append(env.render())

    total_reward = 0
    done = False
    for i in range(env.spec.max_episode_steps):
        # this is where you would insert your policy
        action, extra = agent.act(state)

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        new_state, reward, done, truncated, _ = env.step(action)
        if frame_buffer is not None:
            frame_buffer.append(env.render())
        total_reward += reward
        if experiences is not None:
            experiences.append(Experience(state, action, reward, new_state, done, extra))
        logging.debug(f'{env.spec.id} Episode: {episode}:{i}; action: {action}; reward: {reward}; total_reward: {total_reward}; state: {state}; done: {done}; truncated: {truncated}')
        state = new_state

        # If the episode has ended then we can reset to start a new episode
        if done or truncated:
            break
    logging.debug(f'{env.spec.id} Episode: {episode}; terminated: {done} in {i} steps; total reward: {total_reward}')
    return total_reward

def eval_output(e, episode_scores: List[float], verbose: int=0):
    if episode_scores:
        mean_reward = np.mean(episode_scores)
        std_reward = np.std(episode_scores)
        if verbose > 0:
            print(f'Episode {e}: {len(episode_scores)} episodes rewards mean: {mean_reward:.2f}; std: {std_reward:.2f}')
        return mean_reward, std_reward
    else:
        return 0, 0

def run_agent(env, n_episode: int, evaluation_episode: int, agent, train: bool, score: float, out_directory: Optional[str]=None, verbose=0):
    verbose = max(verbose, 0 if train else 1)
    scores_deque = collections.deque(maxlen=evaluation_episode)
    experiences: collections.deque[Experience] = collections.deque(maxlen=MAX_EXPERIENCES_SIZE)
    frame_buffer: List[np.ndarray] = []
    try:
        for e in range(1, n_episode + 1):
            reward = play_episode(env, e, agent, experiences if train else None, frame_buffer if out_directory else None)
            scores_deque.append(reward)
            if len(frame_buffer) and out_directory:
                filename = os.path.join(out_directory, f'{env.spec.id}-episode-{e}.mp4')
                imageio.mimwrite(filename, [np.array(img_frame) for _, img_frame in enumerate(frame_buffer)], fps=env.metadata['render_fps'])
                print(f"Video saved to {filename}")
                frame_buffer.clear()
            if train and experiences:
                logging.info(f'{env.spec.id} Episode: {e} training agent with {len(experiences)} experiences')
                agent.reinforce(experiences)

            if e % evaluation_episode == 0:
                mean_score, std_score = eval_output(e, list(scores_deque), verbose=verbose)
                if mean_score - 2 * std_score >= score:
                    print(f"Stopping training at episode {e} with mean score: {mean_score:.2f}, std: {std_score:.2f}")
                    break
                scores_deque.clear()

        if scores_deque:
            eval_output(e, list(scores_deque), verbose=verbose)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")

def main(argv: List[str]):
    parser = argparse.ArgumentParser(description='Train and play gymnasium environment.')
    parser.add_argument('-v', '--verbose', action="count", default=0, help="Increase verbosity level")
    parser.add_argument('-e', '--envId', help='environment Id')
    parser.add_argument('-a', '--agent', default='RandomPolicy', help='agent module name')
    parser.add_argument('-f', '--filename', type=str, help='filename load the previous training state')
    parser.add_argument('-n', '--num-episode', type=int, help='number of episode')
    parser.add_argument('-m', '--max-step', type=int, help='max number of step in an episode')
    parser.add_argument('-u', '--evaluation-episode', type=int, help='evaluation episode')
    parser.add_argument('-t', '--train', action='store_true', help='train the policy')
    parser.add_argument('-s', '--score', type=float, default=float('inf'), help='score threshold to stop training')
    parser.add_argument('-r', '--render', help='render mode display or video output folder')
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
    
    max_step = args.max_step or state.get('max_step', None)

    # Initialise the environment
    env = gym.make(envId, max_episode_steps=max_step, render_mode=render_mode)

    agent_module = state.get('agent', args.agent)
    try:
        pm = importlib.import_module(agent_module)
    except ImportError:
        print(f"Policy module {agent_module} not found.", file=sys.stderr)
        exit(1)

    agent = pm.load_agent(env, state) if state else pm.create_agent(env, unrecognized)

    n_episode = args.num_episode or (1000 if args.train else 10)
 
    max_step_env = env.spec.max_episode_steps or max_step # type: ignore
    if args.train:
        print(f'train {agent_module} for {envId} for {n_episode} episodes with {max_step_env} steps')
        eval_episode = args.evaluation_episode or int(n_episode / 25) or n_episode
    else:
        print(f'eval {agent_module} for {envId} for {n_episode} episodes with {max_step_env} steps')
        eval_episode = n_episode

    run_agent(env, n_episode, eval_episode, agent, train=args.train, score=args.score, verbose=args.verbose, out_directory=out_directory)
    if args.train:
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