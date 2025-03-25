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
import pickle
def get_args():
    parser = argparse.ArgumentParser()
    add_domain_args(parser)
    add_scheduler_args(parser)
    add_llm_args(parser)
    parser.add_argument('--data_index', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument("--sglang", action=argparse.BooleanOptionalAction,default=True, help="use sglang")
    parser.add_argument('--n_gpu', type=int, default=1, help="Number of GPUs to use")
    parser.add_argument('--llm_model', type=str, default='/home/flowers/work/hf/Qwen2.5-Coder-3B-Instruct', help="LLM model name")
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
    list_problem_to_solve = [problem_id for problem_id in range(len(list_all_actions))]
    list_solved_history = []
    all_metrics = [[] for _ in range(len(list_all_actions))]
    for problem_id in range(len(list_all_actions)):
        list_all_actions[problem_id] = [new_action(act) for act in list_all_actions[problem_id]]

    max_steps = 300

    path_save_res = os.path.join(result_dir, arg2name(args)+'_response.pkl')
    print("path_save_res", path_save_res)
    if os.path.exists(path_save_res):
        print("loading previous response")
        with open(path_save_res, "rb") as f:
            list_save_response = pickle.load(f)
    else:
        list_save_response = {str(pb_id):[] for pb_id in list_problem_to_solve}
    for pb_id in list_problem_to_solve:
        if str(pb_id) not in list_save_response:
            list_save_response[str(pb_id)] = []
    for si in trange(max_steps, desc="steps"):
        list_save_response_step = {str(pb_id):{} for pb_id in list_problem_to_solve}
        list_prompts = []
        list_idx_actions_selected = {}
        for problem_id in list_problem_to_solve:
            actions = list_all_actions[problem_id]
            action = max(actions, key=lambda a: rng.beta(a['alpha'], a['beta']))
            # action_id 
            list_idx_actions_selected[problem_id] = action["index"]
            prompt = domain.step_get_prompt(list_idx_actions_selected[problem_id],problem_id)
            list_prompts.append(prompt)
            list_save_response_step[str(problem_id)]["prompt"] = prompt
        # break

        list_response = [[] for _ in range(len(list_prompts))]
        list_id_cache_flag = []
        # check if prompt is already in the list_save_response so we can skip it and load it form there
        list_prompt_to_gen = []
        list_id_prompt_togen_to_idx = []
        for idx, problem_id in enumerate(list_problem_to_solve):
            flag_found = False
            if len(list_save_response[str(problem_id)]) > si and not flag_found:
                if list_save_response[str(problem_id)][si]["prompt"][1]["content"] == list_prompts[idx][1]["content"]:
                    list_response[idx].append(list_save_response[str(problem_id)][si]["response"])
                    list_id_cache_flag.append(idx)
                    flag_found = True
                    
            if not flag_found:
                list_prompt_to_gen.append(list_prompts[idx])
                list_id_prompt_togen_to_idx.append(idx)
                    

        
        # for idx, problem_id in enumerate(list_problem_to_solve):
        #     if idx not in list_id_cache_flag:
        #         list_prompt_to_gen.append(list_prompts[idx])
        #         list_id_prompt_togen_to_idx.append(idx)

        # generate solutions
        print("==="*10)
        print("generating response")
        # check if prompt is already in the list_save_response so we can skip it and load it form there
        print(f"generating {len(list_prompt_to_gen)} responses")
        list_response_gen = llm_serv.generate(list_prompt_to_gen)
        assert len(list_response_gen) == len(list_id_prompt_togen_to_idx)
        for idx in range(len(list_response_gen)):
            list_response[list_id_prompt_togen_to_idx[idx]] = list_response_gen[idx]

        # add here responses

        # compute rewards
        list_solved = []
        print("checking response")
        for idx,problem_id in enumerate(tqdm(list_problem_to_solve)):
            list_save_response_step[str(problem_id)]["response"] = list_response[idx][0]
            if idx in list_id_cache_flag:   
                # if cache found
                result = list_save_response[str(problem_id)][si]["result"]
                reward, done, new_actions = domain.step_execute_cache(list_idx_actions_selected[problem_id],problem_id,list_response[idx][0],result,return_res=False)
            else:
                reward, done, new_actions, result = domain.step_execute(list_idx_actions_selected[problem_id],problem_id,list_response[idx][0],return_res=True)

            list_save_response_step[str(problem_id)]["reward"] = reward
            list_save_response_step[str(problem_id)]["done"] = done
            list_save_response_step[str(problem_id)]["result"] = result
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
            if len(list_save_response[key]) <= si:
                list_save_response[key].append({})
            # list_save_response[key].append(list_save_response_step[key])
            list_save_response[key][si] = list_save_response_step[key]

        with open(path_save_res, 'wb') as f:
            pickle.dump(list_save_response, f)
    for problem_id in range(len(list_all_actions)):
        all_metrics[problem_id] += [all_metrics[problem_id][-1]] * (max_steps - len(all_metrics[problem_id]))



        # print(domain.summarize_results(metrics))
    with open(os.path.join(result_dir, arg2name(args)+'.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)

    if args.sglang:
        llm_serv.terminate()
if __name__ == '__main__':
    main()
