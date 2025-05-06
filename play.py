#! /usr/bin/env python3

import argparse
import collections
import datetime
import gymnasium as gym
import importlib
import imageio
import numpy as np
import os
import torch

def play_episode(env, episode, policy, train=False, verbose=0, render=False):
    frame_buffer = []
    replay_buffer = []
    policy.train(train)

    state, info = env.reset()
    if render:
        frame_buffer.append(env.render())

    total_reward = 0
    terminated = False
    for i in range(env.spec.max_episode_steps):
        # this is where you would insert your policy
        action, extra = policy.act(state)

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        new_state, reward, terminated, truncated, info = env.step(action)
        if render:
            frame_buffer.append(env.render())
        total_reward += reward
        if train:
            replay_buffer.append((state, action, reward, new_state, extra))
        state = new_state

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()
            break
    if train:
        policy.reinforce(replay_buffer)
    if verbose > 1:
        print(f'{env.spec.id} Episode: {episode}; terminated: {terminated} in {i} steps; total reward: {total_reward}')
    return total_reward, frame_buffer

def eval_output(e, episode_scores):
    if episode_scores:
        mean_reward = np.mean(episode_scores)
        std_reward = np.std(episode_scores)
        print(f'Episode {e}: {len(episode_scores)} episode rewards mean: {mean_reward:.2f}; std: {std_reward:.2f}')

def run_agent(env, n_episode, evaluation_episode, policy, train, out_directory=None, verbose=0):
    scores_deque = collections.deque(maxlen=evaluation_episode)
    try:
        for e in range(1, n_episode + 1):
            reward, images = play_episode(env, e, policy, train=train, verbose=verbose, render=out_directory!=None)
            scores_deque.append(reward)
            if images and out_directory:
                filename = os.path.join(out_directory, f'{env.spec.id}-episode-{e}.mp4')
                imageio.mimwrite(filename, [np.array(img) for i, img in enumerate(images)], format='mp4', fps=env.metadata['render_fps'])
                print(f"Video saved to {filename}")

            if e % evaluation_episode == 0:
                eval_output(e, scores_deque)
                scores_deque.clear()

        if scores_deque:
            eval_output(e, scores_deque)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and play gymnasium environment.')
    parser.add_argument('-v', '--verbose', action="count", default=0, help="Increase verbosity level")
    parser.add_argument('-e', '--envId', help='environment Id')
    parser.add_argument('-n', '--num-episode', type=int, help='number of episode')
    parser.add_argument('-m', '--max-step', type=int, default=1000, help='max number of step in an episode')
    parser.add_argument('-u', '--evaluation-episode', type=int, help='evaluation episode')
    parser.add_argument('-p', '--policy', default='RandomPolicy', help='policy module name')
    parser.add_argument('-t', '--train', action='store_true', help='train the policy')
    parser.add_argument('-r', '--render', help='render mode display or video output folder')
    parser.add_argument('-f', '--filename', type=str, help='filename to save or load the policy')

    args, unrecognized = parser.parse_known_args()

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
        state = torch.load(args.filename)
    
    envId = state['env'] if 'env' in state else args.envId
    if not envId:
        print("Please provide an environment ID or a filename to load a environment.")
        parser.print_help()
        exit(0)

    # Initialise the environment
    env = gym.make(envId, max_episode_steps=args.max_step, render_mode=render_mode)

    policyName = state['policy'] if 'policy' in state else args.policy
    try:
        pm = importlib.import_module(policyName)
    except ImportError:
        print(f"Policy module {args.policy} not found.")
        exit(1)

    policy = pm.load_policy(env, state, verbose=args.verbose) if state else pm.create_policy(env, unrecognized, verbose=args.verbose)

    n_episode = args.num_episode or (1000 if args.train else 10)
 
    if args.train:
        print(f'train {args.policy} for {envId} for {n_episode} episodes')
        eval_episode = args.evaluation_episode or int(n_episode / 25) or n_episode
    else:
        print(f'eval {args.policy} for {envId} ro {n_episode} episodes')
        eval_episode = n_episode

    run_agent(env, n_episode, eval_episode, policy, train=args.train, verbose=args.verbose, out_directory=out_directory)
    if args.train:
        state = policy.get_state_dict()
        state['env'] = envId
        state['policy'] = args.policy
        filename = args.filename or f'{envId}-{args.policy}-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pt'
        torch.save(state, filename)
        print(f"Policy saved to {filename}")

    env.close()
