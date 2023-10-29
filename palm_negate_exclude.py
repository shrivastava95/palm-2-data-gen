import os
import sys
import json
import time
import torch
from tqdm import tqdm
from random import randint
from dotenv import load_dotenv
import google.generativeai as palm

sys.path.append(os.path.abspath('./utils'))
from config import defaults
from generation import generate_result
sys.path.append(os.path.abspath('./sop_utils'))
import negation_exclusion

buffer_time = 1.5 # minimum time between calls
maximum_retries = 5 # max number of times a prompt is retried
prev_time = time.time() - buffer_time # the last time the api was called

load_dotenv()
palm.configure(api_key=os.getenv('PALM_API_KEY'))


if __name__ == "__main__":
    save_path = 'sop_dec_neg_exc.pt'
    if os.path.exists(save_path):
        save = torch.load(save_path)
    else:
        save = {
            'sentences':[],
            'sop_decompositions':[],
            'sop_chosen-objects':[],
            'sop_negations':[],
            'sop_exclusions':[],
        }
        torch.save(save, save_path)
    sop_decompositions = torch.load('sop_decompositions.pt')
    
    bar = tqdm(total=len(sop_decompositions['sentences']))
    prev_time = time.time() - buffer_time
    for idx, (sentence, sop_decomposition) in enumerate(
        list(zip(
            sop_decompositions['sentences'],
            sop_decompositions['sop_decompositions']))):
        if idx < len(save['sentences']):
            bar.update(1)
            continue
        #######
        objects = [op['object'] for op in sop_decomposition['object-predicate-pairs']]
        chosen_object = objects[randint(0, len(objects)-1)]
        prev_time, result = negation_exclusion.get_negation_exclusion(
            settings=defaults,
            sentence=sentence,
            chosen_object=chosen_object,
            sop_decomposition=sop_decomposition,
            prev_time=prev_time,
            buffer_time=buffer_time,
            maximum_retries=maximum_retries
        )
        #######
        try:
            result_json = json.loads(result)
            negation = result_json['negative-prompt']
            exclusion = result_json['exclusion-prompt']
            save['sentences'].append(sentence)
            save['sop_decompositions'].append(sop_decomposition)
            save['sop_chosen-objects'].append(chosen_object)
            save['sop_negations'].append(negation)
            save['sop_exclusions'].append(exclusion)
        except Exception as e:
            save['sentences'].append(sentence)
            save['sop_decompositions'].append(sop_decomposition)
            save['sop_chosen-objects'].append(chosen_object)
            save['sop_negations'].append(None)
            save['sop_exclusions'].append(None)
            print(e)
        # print('\n'.join([str(idx), sentence, chosen_object, negation, exclusion]))
        bar.update(1)
        torch.save(save, save_path)
    bar.close()