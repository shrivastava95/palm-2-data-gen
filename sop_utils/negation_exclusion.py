import json
import os
import sys

sys.path.append(os.path.abspath('../utils'))
from generation import generate_result

allowed_predicates = ['without', 'not']

instructions = f"""Given a sentence and its decomposition into subject, and all the other object-predicate pairs, along with a 'chosen-object', form two sentences: a 'negative-prompt', and a 'exclusion-prompt'.
The 'negative-prompt' should be similar to the original sentence, except that its association to the subject must be negated by replacing or modifying its associated predicate with one of the words: {allowed_predicates}.
The 'exclusion-prompt' should be similar to the original sentence, except that the 'chosen-object' and its associated predicate from the decomposition must be entirely removed from the sentence altogether.
Keep in mind that the predicate forms the link between the object and the subject in a sentence."""

# todo: make one example for modification.
# todo: make two examples for entire replacement.

examples = [
    {
        'input': {
            'subject': 'woman',
            'sentence': 'A woman next to a table in the dining area',
            'object-predicate-pairs': [
                {
                    'object': 'table',
                    'predicate': 'next to'
                },
                {
                    'object': 'dining area',
                    'predicate': 'in the',
                }
            ],
            'chosen-object': 'table'
        },
        'output': {
            'negative-prompt': 'A woman, not next to a table, in the dining area.',
            'exclusion-prompt': 'A woman in the dining area',
        }
    },
    {
        'input': {
            'subject': 'flag',
            'sentence': 'A flag on a dome',
            'object-predicate-pairs': [
                {
                    'object': 'dome',
                    'predicate': 'on a',
                }
            ],
            'chosen-object': 'dome'
        },
        'output': {
            'negative-prompt': 'A flag, not on a dome',
            'exclusion-prompt': 'A flag',
        }
    },
    {
        'input': {
            'subject': 'person',
            'sentence': 'A person on skis makes her way through the snow',
            'object-predicate-pairs': [
                {
                    'object': 'skis',
                    'predicate': 'on'
                },
                {
                    'object': 'snow',
                    'predicate': 'through the',
                }
            ],
            'chosen-object': 'snow'
        },
        'output': {
            'negative-prompt': 'A person on skis without snow',
            'exclusion-prompt': 'A person on skis',
        }
    },
]



def get_negation_exclusion(settings, sentence, chosen_object, sop_decomposition, prev_time, buffer_time, maximum_retries, instructions=instructions, examples=examples):
    prompt = """"""
    prompt += instructions + '\n'

    for example in examples:
        ex_input = json.dumps(example['input'], indent=4)
        prompt += "input: " + ex_input + '\n'
        ex_output = json.dumps(example['output'], indent=4)
        prompt += "output: " + ex_output + '\n'

    qu_input = json.dumps(
        {
            'subject': sop_decomposition['subject'],
            'sentence': sentence,
            'object-predicate-pairs': sop_decomposition['object-predicate-pairs'],
            'chosen-object': chosen_object,
        }, indent=4)
    prompt += "input: " + qu_input + '\n'
    prompt += "output: "
    prev_time, result = generate_result(settings, prompt, prev_time, buffer_time, maximum_retries)
    return prev_time, result
