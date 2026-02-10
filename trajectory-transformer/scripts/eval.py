import json
from os.path import join

import numpy as np

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    beam_plan,
    make_prefix,
    extract_actions,
    update_context,
)


class Parser(utils.Parser):
    dataset: str = 'antmaze-umaze-v0'
    config: str = 'config.offline'
    episodes: int = 5
    seed: int = 0
    render: bool = False


#######################
######## setup ########
#######################

args = Parser().parse_args('plan')

#######################
####### models ########
#######################

dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
        'data_config.pkl')

gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
        epoch=args.gpt_epoch, device=args.device)

#######################
####### dataset #######
#######################

env = datasets.load_environment(args.dataset)
renderer = utils.make_renderer(args) if args.render else None

discretizer = dataset.discretizer
discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

value_fn = lambda x: discretizer.value_fn(x, args.percentile)
preprocess_fn = datasets.get_preprocess_fn(env.name)

q_client = None
if getattr(args, 'use_q_heuristic', False):
    q_client = utils.QClient(args.q_host, args.q_port)
    q_client.connect()

#######################
###### evaluation #####
#######################

scores = []
returns = []
lengths = []

for ep in range(args.episodes):
    # Best-effort seeding across gym versions
    try:
        observation, _ = env.reset(seed=args.seed + ep)
    except Exception:
        try:
            env.seed(args.seed + ep)
        except Exception:
            pass
        observation = env.reset()

    total_reward = 0.0
    context = []
    rollout = [observation.copy()]

    T = env.max_episode_steps
    for t in range(T):
        observation = preprocess_fn(observation)

        if t % args.plan_freq == 0:
            prefix = make_prefix(discretizer, context, observation, args.prefix_context)
            sequence = beam_plan(
                gpt, value_fn, prefix,
                args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                discount, args.max_context_transitions, verbose=args.verbose,
                k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
                discretizer=discretizer,
                use_q_heuristic=args.use_q_heuristic,
                q_client=q_client,
                q_weight=args.q_weight,
            )
        else:
            sequence = sequence[1:]

        sequence_recon = discretizer.reconstruct(sequence)
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)

        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward

        rollout.append(next_observation.copy())
        context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions)

        if renderer is not None and (t % args.vis_freq == 0 or terminal or t == T - 1):
            renderer.render_plan(join(args.savepath, f'ep{ep}_t{t}_plan.mp4'), sequence_recon, env.state_vector())
            renderer.render_rollout(join(args.savepath, f'ep{ep}_rollout.mp4'), rollout, fps=80)

        if terminal:
            break

        observation = next_observation

    score = env.get_normalized_score(total_reward)
    scores.append(score)
    returns.append(total_reward)
    lengths.append(t + 1)

    print(
        f'[ eval ] ep {ep+1}/{args.episodes} | return {total_reward:.2f} | '
        f'score {score:.4f} | length {t+1} | {args.dataset} | {args.exp_name} | {args.suffix}'
    )

if q_client is not None:
    q_client.close()

results = {
    'dataset': args.dataset,
    'gpt_epoch': gpt_epoch,
    'episodes': args.episodes,
    'seed': args.seed,
    'scores': scores,
    'returns': returns,
    'lengths': lengths,
    'mean_score': float(np.mean(scores)) if scores else None,
    'mean_return': float(np.mean(returns)) if returns else None,
}

json_path = join(args.savepath, 'eval.json')
json.dump(results, open(json_path, 'w'), indent=2, sort_keys=True)
print(f'[ eval ] saved: {json_path}')
