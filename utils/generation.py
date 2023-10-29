import os
import sys
import json
import time
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as palm

sys.path.append(os.path.abspath('./utils'))
from config import defaults


def generate_result(settings, prompt, prev_time, buffer_time, maximum_retries):
    for retry in range(maximum_retries):
        try:
            curr_time = time.time()
            time_until_next_api_call = max(0, buffer_time - (curr_time - prev_time))
            time.sleep(time_until_next_api_call)
            
            temp = time.time()
            response = palm.generate_text(
                **defaults,
                prompt=prompt
            )
            result = response.result
            prev_time = temp
            return prev_time, result
        except Exception as e:
            print(e)
            continue
    return prev_time, 'error : maximum retries reached'


if __name__ == "__main__":
    input = """{
    'sentence': 'A person on skis makes her way through the snow'
    }"""
    prompt = f"""Decompose the given sentence into its subject, and all the other object-predicate pairs. Keep in mind that the predicate forms the link between the object and the subject in a sentence.
    input: {{
        'sentence': 'A woman next to a table in the dining area'
    }}
    output: {{
        'subject': 'woman',
        'object-predicate-pairs': [
            {{
                'object': 'table',
                'predicate': 'next to'
            }},
            {{
                'object': 'dining area',
                'predicate': 'in the',
            }}
        ]
    }}
    input: {{
        'sentence': 'A flag on a dome'
    }}
    output: {{
        'subject': 'flag',
        'object-predicate-pairs': [
            {{
                'object': 'dome',
                'predicate': 'on a'
            }}
        ]
    }}
    input: {input}
    output:"""
    prev_time = time.time()
    buffer_time = 3
    maximum_retries = 5
    load_dotenv()
    palm.configure(api_key=os.getenv('PALM_API_KEY'))

    result = generate_result(
        settings=defaults,
        prompt=prompt,
        prev_time=prev_time,
        buffer_time=buffer_time,
        maximum_retries=maximum_retries,
    )
    # with open('sample_output.out', 'w') as f:
    #     f.write(result)