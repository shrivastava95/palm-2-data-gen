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
from generation import generate_result
sys.path.append(os.path.abspath('./sop_utils'))
import decomposition

buffer_time = 1.5 # minimum time between calls
maximum_retries = 5 # max number of times a prompt is retried
prev_time = time.time() - buffer_time # the last time the api was called

load_dotenv()
palm.configure(api_key=os.getenv('PALM_API_KEY'))


if __name__ == "__main__":
    save_path = 'sop_decompositions.pt'
    if os.path.exists(save_path):
        save = torch.load(save_path)
    else:
        save = {'sentences':[], 'sop_decompositions': []}
        torch.save(save, save_path)

    gui_helper_1000_path = 'gui_helper_1000.pt'
    gui_helper_1000 = torch.load(gui_helper_1000_path)

    captions = [sorted(caps, key=lambda x:len(x))[-1]
                for caps in gui_helper_1000['captions']]
    bar = tqdm(total=len(captions))
    prev_time = time.time() - buffer_time
    for idx, caption in enumerate(captions):
        if idx < len(save['sentences']):
            bar.update(1)
            continue
        #######
        prev_time, result = decomposition.get_decomposition(
            settings=defaults,
            sentence=caption,
            prev_time=prev_time,
            buffer_time=buffer_time,
            maximum_retries=maximum_retries
        )
        #######
        print(idx, prev_time, result, caption)
        result_json = json.loads(result)
        save['sentences'].append(caption)
        save['sop_decompositions'].append(result_json)

        print('\n'.join([str(idx), caption, result]))
        bar.update(1)
        torch.save(save, save_path)
    bar.close()