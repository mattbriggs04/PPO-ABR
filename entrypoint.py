"""
Entrypoint for simulator.

ClientMessage class imported from course, authored by Zach Peats.

Authored by Matt Briggs
License: MIT
"""

from typing import List
from pathlib import Path
from collections import deque
import torch
from .network import ACNet

# ======================================================================================================================
# ClientMessage class by Zach Peats
class ClientMessage:
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

	quality_coefficient: float
	variation_coefficient: float
	rebuffering_coefficient: float
# ======================================================================================================================


# observation/state constants (matches ppo.py training)
TP_HIST_LEN = 5
QUALITY_LEVELS = 3 # always 3
OBS_DIM = 2 + TP_HIST_LEN + 2  # = 9 (buf_frac, last_q, tp_hist x5, chunk_dl_time, chunks_remaining)
TP_NORM_SCALAR = 5.0 # self.tp_norm_scalar in ppo.py

# load trained model
_device = torch.device("cpu")
_model = ACNet(obs_dim=OBS_DIM, act_dim=QUALITY_LEVELS)
_model_path = Path("./ppo_model.pt")
_model.load_state_dict(torch.load(_model_path, map_location=_device, weights_only=True))
_model.eval()

# per-episode state (reset on reload between tests)
_prev_quality = 0
_tp_hist = deque([0.0] * TP_HIST_LEN, maxlen=TP_HIST_LEN)
_chunk_idx = 0
_total_chunks = None  # set on first call


def _message_to_tensor(msg: ClientMessage, chunks_remaining: float) -> torch.Tensor:
	# matches ppo.py _message_to_tensor (can't import PPO directly due to circular imports)
	buf_frac = msg.buffer_seconds_until_empty / max(msg.buffer_max_size, 1e-6)
	last_q = _prev_quality / max(msg.quality_levels - 1, 1)
	tp_hist_norm = [x / TP_NORM_SCALAR for x in list(_tp_hist)]

	# estimated download time for base quality chunk, normalized to [0, 1]
	mean_tp = sum(_tp_hist) / max(sum(1 for x in _tp_hist if x > 0), 1)
	if mean_tp > 0:
		chunk_dl_time = min(msg.quality_bitrates[0] / mean_tp, 10.0) / 10.0
	else:
		chunk_dl_time = 1.0  # first chunk, no throughput data yet => assume slow

	return torch.tensor([buf_frac, last_q, *tp_hist_norm, chunk_dl_time, chunks_remaining], dtype=torch.float32, device=_device)

def entrypoint(client_message: ClientMessage):
	global _prev_quality, _tp_hist, _chunk_idx, _total_chunks

	# previous_throughput in the message is throughput for the chunk that just finished downloading.
	# ingest it before building the next state so training and inference histories match.
	if client_message.previous_throughput > 0:
		_tp_hist.appendleft(client_message.previous_throughput)

	# figure out total chunks on first call, then track remaining fraction
	if _total_chunks is None:
		_total_chunks = len(client_message.upcoming_quality_bitrates) + 1
	chunks_remaining = (_total_chunks - 1 - _chunk_idx) / max(_total_chunks - 1, 1)

	# build state/observation tensor
	obs = _message_to_tensor(client_message, chunks_remaining)

	# run policy forward pass, then choose action greedily
	with torch.no_grad():
		logits, _ = _model(obs.unsqueeze(0))
		quality = int(logits.argmax(dim=-1).item()) # size is (batch=1, QUALITY_LEVELS=3), choose over quality levels

	# update state for next call
	_prev_quality = quality
	_chunk_idx += 1

	return quality
