
import functools
from typing import Callable, Optional, Union

from brax import envs
from brax.v1 import envs as envs_v1
from brax.training.agents.ppo import train as ppoTrain, networks as ppoNetworks
from brax.training.agents.sac import train as sacTrain, networks as sacNetworks
from brax.training import types

from flax.linen import swish, relu

def getTrainFunc(
	agentType = 'ppo',
	hidden_layer_sizes = (256,) * 2,
	policy_hidden_layer_sizes = (32,) * 4,
	value_hidden_layer_sizes = (256,) * 5,
	activation = 'swish',
	num_timesteps = 100_000_000,
	num_evals = 20,
	reward_scaling = 0.1,
	episode_length = 1000,
	normalize_observations = True,
	action_repeat = 1,
	unroll_length = 15,
	num_minibatches = 32,
	num_updates_per_batch = 8,
	discounting = 0.97,
	learning_rate = 6e-4,
	entropy_cost = 1e-2,
	num_envs = 2048,
	batch_size = 1024,
	grad_updates_per_step = 32,
	max_devices_per_host = 1,
	max_replay_size = 1048576,
	min_replay_size = 8192,
	seed = 1):

	if activation == 'relu':
		activationFunc = relu
	else:
		activationFunc = swish

	if agentType == 'sac':
		def networkFunc(
			observation_size: int,
			action_size: int,
			preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor):
			
			return sacNetworks.make_sac_networks(
				observation_size = observation_size,
				action_size = action_size,
				preprocess_observations_fn = preprocess_observations_fn,
				hidden_layer_sizes = hidden_layer_sizes,
				activation = activationFunc)

		def trainFunc(
			environment: Union[envs_v1.Env, envs.Env],
			num_timesteps,
			episode_length: int,
			action_repeat: int = 1,
			num_envs: int = 1,
			num_eval_envs: int = 128,
			learning_rate: float = 1e-4,
			discounting: float = 0.9,
			seed: int = 0,
			batch_size: int = 256,
			num_evals: int = 1,
			normalize_observations: bool = False,
			max_devices_per_host: Optional[int] = None,
			reward_scaling: float = 1.,
			tau: float = 0.005,
			min_replay_size: int = 0,
			max_replay_size: Optional[int] = None,
			grad_updates_per_step: int = 1,
			deterministic_eval: bool = False,
			progress_fn: Callable[[int, types.Metrics], None] = lambda *args: None,
			checkpoint_logdir: Optional[str] = None,
			eval_env: Optional[envs.Env] = None):

			return sacTrain.train(
				environment = environment,
				num_timesteps = num_timesteps,
				episode_length = episode_length,
				action_repeat = action_repeat,
				num_envs = num_envs,
				num_eval_envs = num_eval_envs,
				learning_rate = learning_rate,
				discounting = discounting,
				seed = seed,
				batch_size = batch_size,
				num_evals = num_evals,
				normalize_observations = normalize_observations,
				max_devices_per_host = max_devices_per_host,
				reward_scaling = reward_scaling,
				tau = tau,
				min_replay_size = min_replay_size,
				max_replay_size = max_replay_size,
				grad_updates_per_step = grad_updates_per_step,
				deterministic_eval = deterministic_eval,
				network_factory = networkFunc,
				progress_fn = progress_fn,
				checkpoint_logdir = checkpoint_logdir,
				eval_env = eval_env)

		return functools.partial(
			trainFunc,
			num_timesteps = num_timesteps,
			num_evals = num_evals,
			reward_scaling = reward_scaling,
			episode_length = episode_length,
			normalize_observations = normalize_observations,
			action_repeat = action_repeat,
			discounting = discounting,
			learning_rate = learning_rate,
			num_envs = num_envs,
			batch_size = batch_size,
			grad_updates_per_step = grad_updates_per_step,
			max_devices_per_host = max_devices_per_host,
			max_replay_size = max_replay_size,
			min_replay_size = min_replay_size,
			seed = seed)

	else:
		def networkFunc(
			observation_size: int,
			action_size: int,
			preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor):

			return ppoNetworks.make_ppo_networks(
				observation_size = observation_size,
				action_size = action_size,
				preprocess_observations_fn = preprocess_observations_fn,
				policy_hidden_layer_sizes = policy_hidden_layer_sizes,
				value_hidden_layer_sizes = value_hidden_layer_sizes,
				activation = activationFunc)

		def trainFunc(
			environment: Union[envs_v1.Env, envs.Env],
			num_timesteps: int,
			episode_length: int,
			action_repeat: int = 1,
			num_envs: int = 1,
			max_devices_per_host: Optional[int] = None,
			num_eval_envs: int = 128,
			learning_rate: float = 1e-4,
			entropy_cost: float = 1e-4,
			discounting: float = 0.9,
			seed: int = 0,
			unroll_length: int = 10,
			batch_size: int = 32,
			num_minibatches: int = 16,
			num_updates_per_batch: int = 2,
			num_evals: int = 1,
			normalize_observations: bool = False,
			reward_scaling: float = 1.,
			clipping_epsilon: float = .3,
			gae_lambda: float = .95,
			deterministic_eval: bool = False,
			progress_fn: Callable[[int, types.Metrics], None] = lambda *args: None,
			normalize_advantage: bool = True,
			eval_env: Optional[envs.Env] = None,
			policy_params_fn: Callable[..., None] = lambda *args: None):

			return ppoTrain.train(
				environment = environment,
				num_timesteps = num_timesteps,
				episode_length = episode_length,
				action_repeat = action_repeat,
				num_envs = num_envs,
				max_devices_per_host = max_devices_per_host,
				num_eval_envs = num_eval_envs,
				learning_rate = learning_rate,
				entropy_cost = entropy_cost,
				discounting = discounting,
				seed = seed,
				unroll_length = unroll_length,
				batch_size = batch_size,
				num_minibatches = num_minibatches,
				num_updates_per_batch = num_updates_per_batch,
				num_evals = num_evals,
				normalize_observations = normalize_observations,
				reward_scaling = reward_scaling,
				clipping_epsilon = clipping_epsilon,
				gae_lambda = gae_lambda,
				deterministic_eval = deterministic_eval,
				network_factory = networkFunc,
				progress_fn = progress_fn,
				normalize_advantage = normalize_advantage,
				eval_env = eval_env,
				policy_params_fn = policy_params_fn)

		return functools.partial(
			trainFunc,
			num_timesteps = num_timesteps,
			num_evals = num_evals,
			reward_scaling = reward_scaling,
			episode_length = episode_length,
			normalize_observations = normalize_observations,
			action_repeat = action_repeat,
			unroll_length = unroll_length,
			num_minibatches = num_minibatches,
			num_updates_per_batch = num_updates_per_batch,
			discounting = discounting,
			learning_rate = learning_rate,
			entropy_cost = entropy_cost,
			num_envs = num_envs,
			batch_size = batch_size,
			seed = seed)
