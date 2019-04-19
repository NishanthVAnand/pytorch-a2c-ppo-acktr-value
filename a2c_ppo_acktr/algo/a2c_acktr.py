import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from a2c_ppo_acktr.algo.kfac import KFACOptimizer


class A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 lr_beta=None,
                 reg_beta=None,
                 delib_center=0.5,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.reg_beta = reg_beta

        self.delib_center = delib_center

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)

        self.beta_value_list = []
        self.param_list = []
        for name, param in actor_critic.named_parameters():
            if "base.beta_value_net" in name :
                self.beta_value_list.append(param)
            else:
                self.param_list.append(param)

        else:
            self.optimizer = optim.RMSprop([{'params': self.param_list},
                 {'params': self.beta_value_list, 'lr': lr_beta}], lr, eps=eps, alpha=alpha)

    def update(self, rollouts, eval_prev_value, eval_prev_rew):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        if eval_prev_rew is not None:
            rewards = torch.cat((eval_prev_rew.unsqueeze(0), rollouts.rewards.squeeze(2)))
        else:
            rewards = torch.cat((torch.zeros(1, num_processes), rollouts.rewards.squeeze(2)))

        values, action_log_probs, dist_entropy, _ , eval_prev_value, betas = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1],
            rollouts.recurrent_hidden_states[0],
            rollouts.masks[:-1],
            rollouts.actions,
            eval_prev_value=eval_prev_value,
            eval_prev_rew=rewards)

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        if self.reg_beta > 0:
            target_beta = torch.zeros_like(betas).fill_(self.delib_center)
            delib_loss = F.mse_loss(betas, target_beta)
        else:
            delib_loss = torch.zeros_like(value_loss)

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef + self.reg_beta * delib_loss).backward(retain_graph=True)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item(), eval_prev_value, delib_loss.item(), rewards[-1,:]
