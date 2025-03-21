#!/usr/bin/env python
# coding=utf-8

import os, sys
import argparse
import json

from .domains import add_domain_args, get_domain
from .scheduler import add_scheduler_args, run_with_scheduler
from .utils.llm import add_llm_args, get_llm, LLM_serv
from .utils.logging import set_logger
from .scheduler.rex import add_rex_args, get_rex_args, rex
from tqdm import trange,tqdm
import numpy as np
def get_args():
    parser = argparse.ArgumentParser()
    add_domain_args(parser)
    add_scheduler_args(parser)
    add_llm_args(parser)
    parser.add_argument('--data_index', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument("--sglang", action=argparse.BooleanOptionalAction,default=True, help="use sglang")
    parser.add_argument('--n_gpu', type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    return args

def arg2name(args):
    args_str = ' '.join(sys.argv[1:])
    specified_args = {
        k: v
        for k, v in args.__dict__.items()
        if '-'+k in args_str or '-'+k.replace('_', '-') in args_str
    }
    name = '_'.join(''.join([p[0] for p in k.split('_')])+str(v) for k, v in specified_args.items())
    if "/" in name:
        name = name.replace("/", "_")
    return name

def main():
    args = get_args()
    set_logger(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', arg2name(args)+'.log'))
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', args.domain)
    os.makedirs(result_dir, exist_ok=True)
    domain = get_domain(args)
    print("==="*10)
    print("save_dir", os.path.join(result_dir, arg2name(args)+'.json'))
    print("==="*10)
    if args.sglang:
        llm_serv = LLM_serv(model_path = args.llm_model, seed=args.llm_seed, temperature = args.llm_temperature, fp8 = True, n_gpu = args.n_gpu,max_tokens=3000)
    smoothing = args.rex_smoothing
    constant = args.rex_constant
    def new_action(act):
        action_index, action_name, heuristic_reward = act
        return {
            'index': action_index,
            'name': action_name,
            'heuristic_reward': heuristic_reward,
            'alpha': smoothing if heuristic_reward is None else smoothing + constant * heuristic_reward,
            'beta': smoothing if heuristic_reward is None else smoothing + constant * (1 - heuristic_reward),
        }
    
    rng = np.random.default_rng(args.llm_seed)
    list_all_actions = domain.reset()
    list_all_actions = list_all_actions
    list_problem_to_solve = [problem_id for problem_id in range(len(list_all_actions))]
    list_solved_history = []
    all_metrics = [[] for _ in range(len(list_all_actions))]
    for problem_id in range(len(list_all_actions)):
        list_all_actions[problem_id] = [new_action(act) for act in list_all_actions[problem_id]]

    max_steps = 300
    list_save_response = {pb_id:[]for pb_id in list_problem_to_solve}

    for si in trange(max_steps, desc="steps"):
        list_save_response_step= {pb_id:{}for pb_id in list_problem_to_solve}
        list_prompts = []
        list_idx_actions_selected = {}
        for problem_id in list_problem_to_solve:
            actions = list_all_actions[problem_id]
            action = max(actions, key=lambda a: rng.beta(a['alpha'], a['beta']))
            # action_id 
            list_idx_actions_selected[problem_id] = action["index"]
            prompt = domain.step_get_prompt(list_idx_actions_selected[problem_id],problem_id)
            list_prompts.append(prompt)
        # break
        # generate solutions
        print("==="*10)
        print("generating response")
        
        list_response = llm_serv.generate(list_prompts)
        # compute rewards
        list_solved = []
        print("checking response")
        for id_resp,problem_id in enumerate(tqdm(list_problem_to_solve, desc="checking problem")):
            reward, done, new_actions = domain.step_execute(list_idx_actions_selected[problem_id],problem_id,list_response[id_resp][0])
            all_metrics[problem_id].append(domain.get_metrics(problem_id))
            if done:
                list_solved.append(problem_id)
                list_solved_history.append(problem_id)
                continue
            try:
                list_all_actions[problem_id][list_idx_actions_selected[problem_id]]["alpha"] += reward
                list_all_actions[problem_id][list_idx_actions_selected[problem_id]]["beta"] += (1 - reward)
            except:
                print("problem_id", problem_id)
                print("list_all_actions[problem_id][list_idx_actions_selected[problem_id]]", list_all_actions[problem_id][list_idx_actions_selected[problem_id]])
                print("list_all_actions[problem_id][list_idx_actions_selected[problem_id]]['alpha']", list_all_actions[problem_id][list_idx_actions_selected[problem_id]]["alpha"])
                print("reward", reward)
                raise
            list_all_actions[problem_id].extend([new_action(act) for act in new_actions])
        if len(list_solved) > 0:
            for problem_id in list_solved:
                list_problem_to_solve = [item for item in list_problem_to_solve if item != problem_id]
        print("==="*10)
        print("step", si)
        print(f"problem solved: {len(list_solved_history)}/{len(list_all_actions)}")
        for key in list_save_response_step:
            list_save_response[key].append(list_save_response_step[key])
    for problem_id in range(len(list_all_actions)):
        all_metrics[problem_id] += [all_metrics[problem_id][-1]] * (max_steps - len(all_metrics[problem_id]))



        # print(domain.summarize_results(metrics))
    with open(os.path.join(result_dir, arg2name(args)+'.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)
    with open(os.path.join(result_dir, arg2name(args)+'_response.json'), 'w') as f:
        json.dump(list_save_response, f, indent=4)
    if args.sglang:
        llm_serv.terminate()
if __name__ == '__main__':
    main()
