#!/usr/bin/env python
# coding=utf-8

import copy

from ....utils.llm import LLM
from ..from_jin.openai_utils import construct_initial_prompts as _construct_initial_prompts
from ..from_jin.utility import extract_actual_solution as _extract_actual_solution
from .utils import _message_d

def init_apps(data, llm, verbose_level=2):
    assert isinstance(llm, LLM)

    request_data = copy.deepcopy(data.data)
    # request_data = _construct_initial_prompts([request_data])[0]
    _construct_initial_prompts([request_data])
    prompt = copy.deepcopy(_message_d)
    prompt[-1]['content'] = request_data['prompt']
    if verbose_level >= 2:
        print('Prompt:')
        for p in prompt:
            print('-'*5, f' Role: {p["role"]} ', '-'*20)
            print(p['content'])
        print()

    with llm.track() as costs:
        response = llm(prompt,)
    if verbose_level >= 2:
        print('-'*5, ' Response: ', '-'*20)
        print(response.choices[0].message.content)
        print()

    solution = _extract_actual_solution({request_data['question_name']: request_data}, request_data['question_name'], response.choices[0].message.content,)
    if verbose_level >= 2:
        print('-'*5, ' Extracted Solution: ', '-'*20)
        print(solution)
        print()
    check_result = data.check(solution)
    if verbose_level >= 2:
        print('-'*5, ' Check Result: ', '-'*20)
        print(check_result)
        print()
    success = check_result['success']

    return {
        'success': success,
        'check_result': check_result,
        'response': response,
        'costs': costs.usage,
    }


def init_apps_get_prompt(data, verbose_level=2):

    request_data = copy.deepcopy(data.data)
    # request_data = _construct_initial_prompts([request_data])[0]
    _construct_initial_prompts([request_data])
    prompt = copy.deepcopy(_message_d)
    prompt[-1]['content'] = request_data['prompt']
    if verbose_level >= 2:
        print('Prompt:')
        for p in prompt:
            print('-'*5, f' Role: {p["role"]} ', '-'*20)
            print(p['content'])
        print()
    return prompt



def init_apps_exec(data, response, verbose_level=2):

    request_data = copy.deepcopy(data.data)
    # # request_data = _construct_initial_prompts([request_data])[0]
    # _construct_initial_prompts([request_data])
    # prompt = copy.deepcopy(_message_d)
    # prompt[-1]['content'] = request_data['prompt']
    # if verbose_level >= 2:
    #     print('Prompt:')
    #     for p in prompt:
    #         print('-'*5, f' Role: {p["role"]} ', '-'*20)
    #         print(p['content'])
    #     print()
    # return prompt
        # response = llm(prompt,)
    if verbose_level >= 2:
        print('-'*5, ' Response: ', '-'*20)
        print(response.choices[0].message.content)
        print()

    solution = _extract_actual_solution({request_data['question_name']: request_data}, request_data['question_name'], response,)
    if verbose_level >= 2:
        print('-'*5, ' Extracted Solution: ', '-'*20)
        print(solution)
        print()
    check_result = data.check(solution)
    if verbose_level >= 2:
        print('-'*5, ' Check Result: ', '-'*20)
        print(check_result)
        print()
    success = check_result['success']

    return {
        'success': success,
        'check_result': check_result,
        'response': response,
        'costs': 0,
    }