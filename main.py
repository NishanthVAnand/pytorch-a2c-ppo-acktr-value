from comet_ml import OfflineExperiment, Experiment

import copy
import glob
import os
import time
from collections import deque

import pickle

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot


args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

if args.comet == "offline":
    experiment = OfflineExperiment(project_name="recurrent-value", workspace="nishanthvanand",
    disabled=args.disable_log, offline_directory="../comet_offline",
    parse_args=False)
elif args.comet == "online":
    experiment = Experiment(api_key="tSACzCGFcetSBTapGBKETFARf",
                        project_name="recurrent-value", workspace="nishanthvanand",
                        disabled=args.disable_log,
                        parse_args=False)
else:
    raise ValueError

experiment.log_parameters(vars(args))

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs, is_minigrid = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False, args.num_frame_stack)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy, 'est_beta_value':args.est_beta_value},
        is_minigrid=is_minigrid, use_rew=args.use_reward)

    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               lr_beta=args.lr_beta, reg_beta=args.reg_beta,
                               delib_center=args.delib_center,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    obs = obs + torch.randn_like(obs) * args.noise_obs
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    
    all_rewards_local = deque()
    all_frame_local = deque()
    all_beta_mean_local = deque()
    all_beta_std_local = deque()

    start = time.time()

    prev_value = 0
    eval_prev_value = 0
    prev_rew = torch.zeros(args.num_processes).to(device)
    eval_prev_rew = torch.zeros(args.num_processes).to(device)

    for j in range(num_updates):
        beta_value_list = []

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, prev_value, beta_value = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                        prev_value=prev_value,
                        prev_rew=prev_rew)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            obs = obs + torch.randn_like(obs) * args.noise_obs

            if not args.cuda:
                beta_value_list.append(beta_value.numpy())
            else:
                beta_value_list.append(beta_value.cpu().numpy())

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])

            prev_rew = reward * masks

            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy, eval_prev_value, beta_loss, eval_prev_rew = agent.update(rollouts, eval_prev_value, eval_prev_rew)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]
                
            torch.save(save_model, os.path.join(save_path, args.env_name +"_seed_"+str(args.seed)+"_est_beta_"+str(args.est_beta_value)+"_lr_beta_"+str(args.lr_beta)+"_beta_reg_"+str(args.reg_beta)+"_entropy_"+str(args.entropy_coef)+"_steps_"+str(args.num_steps)+"_noise_obs_"+str(args.noise_obs)+".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))

            experiment.log_metrics({"mean reward": np.mean(episode_rewards),
                                 "Value loss": value_loss, "Action Loss": action_loss,
                                 "Beta loss": beta_loss,
                                 "Beta mean": np.array(beta_value_list).mean(),
                                 "Beta std": np.array(beta_value_list).std()}, 
                                 step=j * args.num_steps * args.num_processes)

            all_rewards_local.append(np.mean(episode_rewards))
            all_frame_local.append(total_num_steps)
            all_beta_mean_local.append(np.array(beta_value_list).mean())
            all_beta_std_local.append(np.array(beta_value_list).std())

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.tensor([[0.0] if done_ else [1.0]
                                           for done_ in done],
                                           dtype=torch.float32,
                                           device=device)
                
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_env_steps)
            except IOError:
                pass

    with open(args.save_local_dir+"Rewards_"+str(args.env_name)+"_seed_"+str(args.seed)+"_est_beta_"+str(args.est_beta_value)+"_lr_beta_"+str(args.lr_beta)+"_beta_reg_"+str(args.reg_beta)+"_entropy_"+str(args.entropy_coef)+"_steps_"+str(args.num_steps)+"_noise_obs_"+str(args.noise_obs)+".pkl", 'wb') as f:
        pickle.dump(all_rewards_local, f)

    with open(args.save_local_dir+"Frames_"+str(args.env_name)+"_seed_"+str(args.seed)+"_est_beta_"+str(args.est_beta_value)+"_lr_beta_"+str(args.lr_beta)+"_beta_reg_"+str(args.reg_beta)+"_entropy_"+str(args.entropy_coef)+"_steps_"+str(args.num_steps)+"_noise_obs_"+str(args.noise_obs)+".pkl", 'wb') as f:
        pickle.dump(all_frame_local, f)

    with open(args.save_local_dir+"beta_mean_"+str(args.env_name)+"_seed_"+str(args.seed)+"_est_beta_"+str(args.est_beta_value)+"_lr_beta_"+str(args.lr_beta)+"_beta_reg_"+str(args.reg_beta)+"_entropy_"+str(args.entropy_coef)+"_steps_"+str(args.num_steps)+"_noise_obs_"+str(args.noise_obs)+".pkl", 'wb') as f:
        pickle.dump(all_beta_mean_local, f)

    with open(args.save_local_dir+"beta_std_"+str(args.env_name)+"_seed_"+str(args.seed)+"_est_beta_"+str(args.est_beta_value)+"_lr_beta_"+str(args.lr_beta)+"_beta_reg_"+str(args.reg_beta)+"_entropy_"+str(args.entropy_coef)+"_steps_"+str(args.num_steps)+"_noise_obs_"+str(args.noise_obs)+".pkl", 'wb') as f:
        pickle.dump(all_beta_std_local, f)

if __name__ == "__main__":
    main()
