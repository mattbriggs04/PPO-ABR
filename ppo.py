import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import deque
from torch.distributions import Categorical

from .network import ACNet
import simulator

from typing import List
class ClientMessage:
	"""
	This class will be filled out and passed to student_entrypoint for your algorithm.
	"""
	total_seconds_elapsed: float	  # The number of simulated seconds elapsed in this test
	previous_throughput: float		  # The measured throughput for the previous chunk in kB/s

	buffer_current_fill: float		    # The number of kB currently in the client buffer
	buffer_seconds_per_chunk: float     # Number of seconds that it takes the client to watch a chunk. Every
										# buffer_seconds_per_chunk, a chunk is consumed from the client buffer.
	buffer_seconds_until_empty: float   # The number of seconds of video left in the client buffer. A chunk must
										# be finished downloading before this time to avoid a rebuffer event.
	buffer_max_size: float              # The maximum size of the client buffer. If the client buffer is filled beyond
										# maximum, then download will be throttled until the buffer is no longer full

	# The quality bitrates are formatted as follows:
	#
	#   quality_levels is an integer reflecting the # of quality levels you may choose from.
	#
	#   quality_bitrates is a list of floats specifying the number of kilobytes the upcoming chunk is at each quality
	#   level. Quality level 2 always costs twice as much as quality level 1, quality level 3 is twice as big as 2, and
	#   so on.
	#       quality_bitrates[0] = kB cost for quality level 1
	#       quality_bitrates[1] = kB cost for quality level 2
	#       ...
	#
	#   upcoming_quality_bitrates is a list of quality_bitrates for future chunks. Each entry is a list of
	#   quality_bitrates that will be used for an upcoming chunk. Use this for algorithms that look forward multiple
	#   chunks in the future. Will shrink and eventually become empty as streaming approaches the end of the video.
	#       upcoming_quality_bitrates[0]: Will be used for quality_bitrates in the next student_entrypoint call
	#       upcoming_quality_bitrates[1]: Will be used for quality_bitrates in the student_entrypoint call after that
	#       ...
	#
	quality_levels: int
	quality_bitrates: List[float]
	upcoming_quality_bitrates: List[List[float]]

	# You may use these to tune your algorithm to each user case! Remember, you can and should change these in the
	# config files to simulate different clients!
	#
	#   User Quality of Experience =    (Average chunk quality) * (Quality Coefficient) +
	#                                   -(Number of changes in chunk quality) * (Variation Coefficient)
	#                                   -(Amount of time spent rebuffering) * (Rebuffering Coefficient)
	#
	#   *QoE is then divided by total number of chunks
	#
	quality_coefficient: float
	variation_coefficient: float
	rebuffering_coefficient: float
# ======================================================================================================================

TP_HIST_LEN = 5
QUALITY_LEVELS = 3  # all .ini configs use 3 quality levels
# obs_dim = (buf_frac, last_q)=2 + tp_hist=5 + chunk_dl_time=1 + chunks_remaining=1 => 9 total
OBS_DIM = 2 + TP_HIST_LEN + 2

# A lot of the boilerplate PPO code is from:
# https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py
class PPO:
    def __init__(self, policy_model: ACNet, device: torch.device, config_dir_path: str):
        self.model = policy_model.to(device)
        self.device = device

        self.lr_start = 3e-4
        self.lr_end = 1e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_start)
        self.crit_loss = nn.MSELoss()

        self.gamma = 0.99 # discount
        self.lmbda = 0.94 # GAE lambda

        self.clip_eps = 0.2
        self.value_clip_eps = 0.2
        self.vf_coef = 0.5
        self.ent_coef_start = 0.02
        self.ent_coef_end = 0.003
        self.max_grad_norm = 0.5
        self.target_kl = 0.03

        self.timesteps_per_batch = 4096
        self.minibatch_size = 256  # split batch into minibatches for better updates

        self.config_paths = sorted(Path(config_dir_path).glob("*.ini"))
        self.tp_norm_scalar = 5.0  # throughputs range roughly 0.2-4.5 Mbps across configs

    def train(self, updates_per_iteration: int, total_timesteps: int, save_path: str):
        t = 0
        iter_idx = 0

        while t < total_timesteps:
            progress = min(t / max(total_timesteps, 1), 1.0)
            curr_lr = self.lr_start + (self.lr_end - self.lr_start) * progress
            curr_ent_coef = self.ent_coef_start + (self.ent_coef_end - self.ent_coef_start) * progress
            for group in self.optimizer.param_groups:
                group["lr"] = curr_lr

            # step 1: rollout
            obs, acts, logp_old, vals_old, rews, dones, lens = self.rollout()
            t += obs.shape[0]
            iter_idx += 1

            # step 2: compute advantages and returns
            adv, returns = self._compute_gae(rews, dones, vals_old)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8) # avoid div by 0

            # step 3: update, backpropagate with minibatches
            self.model.train()
            batch_size = obs.shape[0]
            indices = np.arange(batch_size)

            stop_early = False
            for _ in range(updates_per_iteration):
                # shuffle and iterate through minibatches
                np.random.shuffle(indices)
                for start in range(0, batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_idx = indices[start:end]

                    mb_obs = obs[mb_idx]
                    mb_acts = acts[mb_idx]
                    mb_logp_old = logp_old[mb_idx]
                    mb_adv = adv[mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_vals_old = vals_old[mb_idx]

                    logits, values = self.model(mb_obs)
                    values = values.squeeze(-1)

                    dist = Categorical(logits=logits)
                    logp_new = dist.log_prob(mb_acts)
                    entropy = dist.entropy().mean()

                    # compute clipped surrogate loss
                    ratio = torch.exp(logp_new - mb_logp_old)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    values_clipped = mb_vals_old + (values - mb_vals_old).clamp(-self.value_clip_eps, self.value_clip_eps)
                    value_loss_unclipped = (values - mb_returns).pow(2)
                    value_loss_clipped = (values_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    loss = policy_loss + self.vf_coef * value_loss - curr_ent_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    approx_kl = (mb_logp_old - logp_new).mean().item()
                    if approx_kl > self.target_kl:
                        stop_early = True
                        break
                if stop_early:
                    break

            # logging
            with torch.no_grad():
                avg_return = returns.mean().item()
                avg_rew = rews.mean().item()
                avg_len = float(np.mean(lens))
            print(f"[iter {iter_idx}] steps={t} | avg_ep_len={avg_len:.1f} | avg_return={avg_return:.3f} | avg_rew={avg_rew:.3f} | lr={curr_lr:.6f} | ent={curr_ent_coef:.4f}")

        # save final model
        torch.save(self.model.state_dict(), save_path)
        print("Training complete")

    def rollout(self):
        self.model.eval()

        batch_obs = []
        batch_acts = []
        batch_logp = []
        batch_vals = []
        batch_rews = []
        batch_dones = []
        batch_lens = []

        steps = 0
        cfg_i = 0
        cfg_order = np.random.permutation(len(self.config_paths))

        while steps < self.timesteps_per_batch:
            # circularly go through all the config files in random order
            config_path = self.config_paths[cfg_order[cfg_i % len(self.config_paths)]]
            cfg_i += 1

            trace, logger, buffer, chunk_qualities, chunk_length = simulator.read_test(
                config_path.as_posix(), print_output=False
            )

            # state variables
            current_time = 0.0
            prev_throughput = 0.0
            prev_quality = 0
            tp_hist = deque([0.0] * TP_HIST_LEN, maxlen=TP_HIST_LEN)

            episode_len = 0

            for chunknum in range(len(chunk_qualities)):
                # build ClientMessage (copied from simulator main)
                msg = ClientMessage()
                msg.total_seconds_elapsed = current_time
                msg.previous_throughput = prev_throughput

                msg.buffer_seconds_per_chunk = chunk_length
                msg.buffer_seconds_until_empty = buffer.seconds_left
                msg.buffer_max_size = buffer.client_buffer_size

                msg.quality_levels = len(chunk_qualities[chunknum])
                msg.quality_bitrates = chunk_qualities[chunknum]
                msg.upcoming_quality_bitrates = (
                    chunk_qualities[chunknum + 1:] if chunknum < len(chunk_qualities) - 1 else []
                )

                msg.quality_coefficient = logger.quality_coeff
                msg.rebuffering_coefficient = logger.rebuffer_coeff
                msg.variation_coefficient = logger.switch_coeff

                # chunks remaining as fraction of total episode
                chunks_remaining = (len(chunk_qualities) - 1 - chunknum) / max(len(chunk_qualities) - 1, 1)

                # observation tensor
                obs_t = self._message_to_tensor(msg, prev_quality, tp_hist, chunks_remaining).to(self.device)

                # run policy
                with torch.no_grad():
                    logits, v = self.model(obs_t.unsqueeze(0))
                    dist = Categorical(logits=logits)
                    act = dist.sample()
                    logp = dist.log_prob(act)
                    v = v.squeeze(-1) # (1, 1) -> (1)

                quality = int(act.item())
                chosen_bitrate = chunk_qualities[chunknum][quality]

                # simulate download (same as simulator main)
                time_elapsed = trace.simulate_download_from_time(current_time, chosen_bitrate)
                rebuff_time = buffer.sim_chunk_download(chosen_bitrate, time_elapsed)

                # update variables
                prev_throughput = chosen_bitrate / max(time_elapsed, 1e-8)
                current_time += time_elapsed
                current_time += buffer.wait_until_buffer_is_not_full(False)

                tp_hist.appendleft(prev_throughput)

                variation = abs(quality - prev_quality)
                r = self._compute_reward(msg, quality, variation, rebuff_time)

                done = 1.0 if (chunknum == len(chunk_qualities) - 1) else 0.0

                # store transition
                batch_obs.append(obs_t)
                batch_acts.append(act.squeeze(0).to(self.device))
                batch_logp.append(logp.squeeze(0).to(self.device))
                batch_vals.append(v.squeeze(0).to(self.device))
                batch_rews.append(torch.tensor(r, dtype=torch.float32, device=self.device))
                batch_dones.append(torch.tensor(done, dtype=torch.float32, device=self.device))

                prev_quality = quality
                episode_len += 1
                steps += 1

                if steps >= self.timesteps_per_batch and done == 1.0:
                    break

            batch_lens.append(episode_len)

        # lists -> tensors
        obs = torch.stack(batch_obs).to(self.device)
        acts = torch.stack(batch_acts).long().to(self.device)
        logp_old = torch.stack(batch_logp).float().to(self.device)
        vals_old = torch.stack(batch_vals).float().to(self.device)
        rews = torch.stack(batch_rews).float().to(self.device)
        dones = torch.stack(batch_dones).float().to(self.device)

        return obs, acts, logp_old, vals_old, rews, dones, batch_lens

    # turns client_message into tensor (observation/state vector for input to policy)
    # matches the obs_dim to ACNet
    def _message_to_tensor(self, msg: ClientMessage, prev_quality: int, tp_hist: deque, chunks_remaining: float):
        # buffer fullness percentage
        buf_frac = msg.buffer_seconds_until_empty / max(msg.buffer_max_size, 1e-6)

        # last selected quality normalized
        last_q = prev_quality / max(msg.quality_levels - 1, 1)

        # q = msg.quality_bitrates
        # q_norm = [float(x) / max(q) for x in q]
        # useless: literally just [0.25, 0.5, 1.0] always

        # normalize throughput history to roughly [0, 1]
        tp_hist_norm = [x / self.tp_norm_scalar for x in list(tp_hist)]

        # estimated download time for base quality chunk, normalized to [0, 1]
        vals = [x for x in tp_hist if x > 0]
        mean_tp = sum(vals) / len(vals) if vals else 0.0
        if mean_tp > 0:
            chunk_dl_time = min(msg.quality_bitrates[0] / mean_tp, 10.0) / 10.0
        else:
            chunk_dl_time = 1.0  # first chunk, no throughput data yet => assume slow
        obs = torch.tensor([buf_frac, last_q, *tp_hist_norm, chunk_dl_time, chunks_remaining], dtype=torch.float32, device=self.device)
        return obs

    def _compute_reward(self, msg: ClientMessage, quality: int, variation: int, rebuff_time: float) -> float:
        return float(msg.quality_coefficient * quality - msg.variation_coefficient * variation - msg.rebuffering_coefficient * rebuff_time)

    def _compute_gae(self, rews: torch.Tensor, dones: torch.Tensor, vals: torch.Tensor):
        # implementation based on https://nn.labml.ai/rl/ppo/gae.html
        T = rews.shape[0]
        adv = torch.zeros(T, dtype=torch.float32, device=self.device)

        last_adv = torch.tensor(0.0, device=self.device)
        last_value = torch.tensor(0.0, device=self.device) # this is V(s_{t+1}) during recursion

        for t in reversed(range(T)):
            mask = 1.0 - dones[t] # 0 if terminal, 1 otherwise
            last_value = last_value * mask
            last_adv = last_adv * mask

            delta = rews[t] + self.gamma * last_value - vals[t]
            last_adv = delta + self.gamma * self.lmbda * last_adv

            adv[t] = last_adv
            last_value = vals[t] # shift: current V becomes next-step V in the backward loop

        returns = adv + vals
        return adv, returns
