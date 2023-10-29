import json
import os
import sys

sys.path.append(os.path.abspath('../utils'))
from generation import generate_result

instructions = """Decompose the given sentence into its subject, and all the other object-predicate pairs. Keep in mind that the predicate forms the link between the object and the subject in a sentence."""

examples = [
    {
        'input': {
            'sentence': 'A woman next to a table in the dining area'
        },
        'output': {
            'subject': 'woman',
            'object-predicate-pairs': [
                {
                    'object': 'table',
                    'predicate': 'next to'
                },
                {
                    'object': 'dining area',
                    'predicate': 'in the',
                }
            ]
        }
    },
    {
        'input': {
            'sentence': 'A flag on a dome'
        },
        'output': {
            'subject': 'flag',
            'object-predicate-pairs': [
                {
                    'object': 'dome',
                    'predicate': 'on a'
                }
            ]
        }
    },
    {
        'input': {
            'sentence': 'A person on skis makes her way through the snow'
        },
        'output': {
            'subject': 'person',
            'object-predicate-pairs': [
                {
                    'object': 'skis',
                    'predicate': 'on'
                },
                {
                    'object': 'snow',
                    'predicate': 'through the'
                }
            ]
        }
    }
]


def get_decomposition(settings, sentence, prev_time, buffer_time, maximum_retries, instructions=instructions, examples=examples):
    prompt = """"""
    prompt += instructions + '\n'

    for example in examples:
        ex_input = json.dumps(example['input'], indent=4)
        prompt += "input: " + ex_input + '\n'
        ex_output = json.dumps(example['output'], indent=4)
        prompt += "output: " + ex_output + '\n'

    qu_input = json.dumps({'sentence': sentence}, indent=4)
    prompt += "input: " + qu_input + '\n'
    prompt += "output: "
    prev_time, result = generate_result(settings, prompt, prev_time, buffer_time, maximum_retries)
    return prev_time, result
