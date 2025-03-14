#!/usr/bin/env python
# coding=utf-8

import copy

from ..base import _Domain
from .data import add_apps_data_args, get_apps_dataset
from .llm_actions.init import init_apps, init_apps_get_prompt, init_apps_exec
from .llm_actions.refine import refine_apps, refine_apps_get_prompt, refine_apps_exec
from ...utils.llm import get_llm
from ...utils.llm.utils import count_tokens_for_openai

def add_apps_args(parser):
    add_apps_data_args(parser)

class InitAPPS:
    def __init__(self, data, llm):
        self.data = data
        self.llm = llm
        self.name = f"InitAPPS({data})"
    def run(self):
        return init_apps(self.data, self.llm)

    def __hash__(self):
        return hash(str(self.data))
    def __eq__(self, other):
        if not isinstance(other, InitAPPS):
            return False
        return hash(self) == hash(other)
    
class InitAPPS_multi:
    def __init__(self, data, verbose_level=0):
        self.data = data
        self.verbose_level = verbose_level
        self.name = f"InitAPPS({data})"
    
    def get_prompt(self):
        return init_apps_get_prompt(self.data, self.verbose_level)
    
    def execute(self,response):
        return init_apps_exec(self.data, response, self.verbose_level)
    
    def __hash__(self):
        return hash(str(self.data))
    def __eq__(self, other):
        if not isinstance(other, InitAPPS):
            return False
        return hash(self) == hash(other)

class RefineAPPS:
    def __init__(self, data, check_result, llm):
        self.data = data
        self.check_result = check_result
        self.llm = llm
        self.name = f"RefineAPPS({check_result['success_rate']}, {len(check_result['solution'])})"
    def run(self):
        return refine_apps(self.data, self.check_result, self.llm)
    def __hash__(self):
        return hash(str(self.data) + '\n\n' + self.check_result['solution'])
    def __eq__(self, other):
        if not isinstance(other, RefineAPPS):
            return False
        return hash(self) == hash(other)
    
class RefineAPPS_multi:
    def __init__(self, data, check_result):
        self.data = data
        self.check_result = check_result
        self.name = f"RefineAPPS({check_result['success_rate']}, {len(check_result['solution'])})"
    def run(self):
        return refine_apps(self.data, self.check_result)
    def get_prompt(self):
        return refine_apps_get_prompt(self.data, self.check_result)
    
    def execute(self,response):
        return refine_apps_exec(self.data, self.check_result, response)
    
    def __hash__(self):
        return hash(str(self.data) + '\n\n' + self.check_result['solution'])
    def __eq__(self, other):
        if not isinstance(other, RefineAPPS):
            return False
        return hash(self) == hash(other)

class APPSDomain(_Domain):
    def __init__(
        self, args, verbose=True,
    ):
        self.llm = get_llm(args)
        self.dataset = get_apps_dataset(args)
        self.verbose = verbose

    def reset(self, problem_id):
        self.problem_id = problem_id
        self.data = self.dataset[problem_id]
        self.actions = [InitAPPS(self.data, self.llm),]

        self.max_metrics = {
            'success': False,
            'success_in_steps': None,
            'success_rate': 0,
        }
        self.cur_step = 0
        output_new_actions = [
            (i, action.name, None)
            for i, action in enumerate(self.actions)
        ]

        if self.verbose:
            print(f"Problem {problem_id} reset")
        return output_new_actions
    
    def step(self, action_index, return_heuristic=False):
        assert 0 <= action_index < len(self.actions)
        self.cur_step += 1
        action = self.actions[action_index]
        print()
        print('='*10, f'Step {self.cur_step}', f'Action {action.name}', '='*10)
        result = action.run()
        reward, done = self.compute_reward(result)
        heuristic = self.compute_heuristic(result)
        new_actions = self.get_new_actions(result)
        self.update_max_metrics(result)
        self.actions += new_actions
        output_new_actions = [
            (len(self.actions) - len(new_actions) + i, new_action.name, heuristic)
            for i, new_action in enumerate(new_actions)
        ]
        if self.verbose:
            print(f"Step {self.cur_step}: {action.name} -> {result['success']}, {reward}, {done}")
            print(f"New actions: {output_new_actions}")
        if not return_heuristic:
            return reward, done, output_new_actions
        else:
            return reward, done, output_new_actions, heuristic
    def update_max_metrics(self, result):
        check_result = result['check_result']
        if check_result['success']:
            self.max_metrics['success'] = True
            self.max_metrics['success_in_steps'] = self.cur_step
        self.max_metrics['success_rate'] = max(
            self.max_metrics['success_rate'],
            self._pass_rate(check_result),
        )
    def get_new_actions(self, result):
        if result['success']:
            return []
        check_result = result['check_result']
        if check_result['solution'] is None:
            return []
        new_action = RefineAPPS(self.data, check_result, self.llm)
        if new_action in self.actions:
            return []
        return [new_action]
    def compute_reward(self, result):
        if result['success']:
            return 1, True
        return 0, False
    def compute_heuristic(self, result):
        check_result = result['check_result']
        return self._pass_rate(check_result)
    def get_metrics(self,):
        return copy.deepcopy(self.max_metrics)
    def _pass_rate(self, check_result):
        return check_result['success_rate']
    def set_seed(self, seed):
        self.llm.set_seed(seed)
    def __len__(self):
        return len(self.dataset)
    def __str__(self,):
        return f"APPSDomain({len(self.dataset)})"



class APPSDomain_multi(_Domain):
    def __init__(
        self, args, verbose=True,
    ):  
        
        self.dataset = get_apps_dataset(args)
        self.list_problem_id = [i for i in range(len(self.dataset))]
        self.verbose = verbose

    def reset(self):
        self.list_data = []
        self.list_actions = []
        self.list_max_metrics = []
        self.list_cur_step = []
        list_output_new_actions = []
        for problem_id in self.list_problem_id:
            self.list_data.append(self.dataset[problem_id])
            self.list_actions.append([InitAPPS_multi(self.dataset[problem_id])])
            self.list_max_metrics.append({
                'success': False,
                'success_in_steps': None,
                'success_rate': 0,
            })
            self.list_cur_step.append(0)
            
            output_new_actions = [
                (i, action.name, None)
                for i, action in enumerate(self.list_actions[-1])
            ]
            list_output_new_actions.append(output_new_actions)

        if self.verbose:
            print(f"All Problem reset")
        return list_output_new_actions
    
    # def step(self, action_index, return_heuristic=False):
    #     assert 0 <= action_index < len(self.actions)
    #     self.cur_step += 1
    #     action = self.actions[action_index]
    #     print()
    #     print('='*10, f'Step {self.cur_step}', f'Action {action.name}', '='*10)
    #     result = action.run()
    #     reward, done = self.compute_reward(result)
    #     heuristic = self.compute_heuristic(result)
    #     new_actions = self.get_new_actions(result)
    #     self.update_max_metrics(result)
    #     self.actions += new_actions
    #     output_new_actions = [
    #         (len(self.actions) - len(new_actions) + i, new_action.name, heuristic)
    #         for i, new_action in enumerate(new_actions)
    #     ]
    #     if self.verbose:
    #         print(f"Step {self.cur_step}: {action.name} -> {result['success']}, {reward}, {done}")
    #         print(f"New actions: {output_new_actions}")
    #     if not return_heuristic:
    #         return reward, done, output_new_actions
    #     else:
    #         return reward, done, output_new_actions, heuristic
        
    def step_get_prompt(self, action_index, problem_id,return_heuristic=False):
        assert 0 <= action_index < len(self.list_actions[problem_id])
        self.list_cur_step[problem_id] += 1
        action = self.list_actions[problem_id][action_index]
        # print()
        # print('='*10, f'Step {self.cur_step}', f'Action {action.name}', '='*10)
        prompt = action.get_prompt()
        return prompt
    
    def step_execute(self, action_index, problem_id, response, return_heuristic=False):
        assert 0 <= action_index < len(self.list_actions[problem_id])
        action = self.list_actions[problem_id][action_index]
        result = action.execute(response)
        reward, done = self.compute_reward(result)
        heuristic = self.compute_heuristic(result)
        new_actions = self.get_new_actions(result,problem_id)
        self.update_max_metrics(result, problem_id)
        self.list_actions[problem_id] += new_actions
        output_new_actions = [
            (len(self.list_actions[problem_id]) - len(new_actions) + i, new_action.name, heuristic)
            for i, new_action in enumerate(new_actions)
        ]
        # if self.verbose:
        #     print(f"Step {self.cur_step}: {action.name} -> {result['success']}, {reward}, {done}")
        #     print(f"New actions: {output_new_actions}")
        if not return_heuristic:
            return reward, done, output_new_actions
        else:
            return reward, done, output_new_actions, heuristic


    def update_max_metrics(self, result, problem_id):
        check_result = result['check_result']
        if check_result['success']:
            self.list_max_metrics[problem_id]['success'] = True
            self.list_max_metrics[problem_id]['success_in_steps'] = self.list_cur_step[problem_id] 
        self.list_max_metrics[problem_id]['success_rate'] = max(
            self.list_max_metrics[problem_id]['success_rate'],
            self._pass_rate(check_result),
        )
    def get_new_actions(self, result, problem_id):
        if result['success']:
            return []
        check_result = result['check_result']
        if check_result['solution'] is None:
            return []
        new_action = RefineAPPS_multi(self.dataset[problem_id], check_result)
        if new_action in self.list_actions[problem_id]:
            return []
        return [new_action]
    def compute_reward(self, result):
        if result['success']:
            return 1, True
        return 0, False
    def compute_heuristic(self, result):
        check_result = result['check_result']
        return self._pass_rate(check_result)
    def get_metrics(self,problem_id):
        return copy.deepcopy(self.list_max_metrics[problem_id])
    def _pass_rate(self, check_result):
        return check_result['success_rate']
    def set_seed(self, seed):
        pass
    def __len__(self):
        return len(self.dataset)
    def __str__(self,):
        return f"APPSDomain({len(self.dataset)})"