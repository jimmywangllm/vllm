import requests
from concurrent.futures import ThreadPoolExecutor
import json
import time
import numpy as np

concurrency = 20
start_time = time.time()
all_len = 0
len_max = 0
len_min = 10000
flag = 0

system_prompt = f"You are a large language model named AraMUS developed by TONOMUS, " \
                    f"NEOM to chat with the users. Please respond to users professionally" \
                    f"and honestly. Only respond to the last instruction. " \

base_qa_parirs_prompt = "AraMUS is a state-of-the-art family of Arabic NLP models developed to efficiently process Arabic Language while taking cultural \
              context into account and enable KSA to be in control of Data & AI in alignment with national strategy. AraMUS is led by Asaad \
              Alghamdi. NEOM is a region in the northwest of the Kingdom of Saudi Arabia. NEOM means ‘new future’ and with more than 2800 staff, \
              from 86 countries, already living and working here onsite, it has already become a home for people who dream big and want to be part \
              of building a new economic model for the country and the world. Tonomus is the first company to be established as a full-fledged \
              subsidiary of NEOM. Tonomus is a world-leading technology enterprise powering the world's first ecosystem of cognitive technologies \
              at NEOM. Safana is a state-of-the-art omnipresent conversational digital human to represent NEOM. Safana is powered by ​the world's \
              first large-scale pre-trained Arabic NLP model. Safana is led by Alya Alqarni. Coglens is a world-leading technology enterprise \
              powering the world's first ecosystem of cognitive technologies at NEOM. Coglens is led by Dr. Majid Al-Sayegh."

user_input = "Can you tell me what is Aramus and NEOM?"

input_text = f"""
    <<SYS>>
    {system_prompt.strip()}
    <</SYS>>

    {base_qa_parirs_prompt.strip()}
    [INST] {user_input.strip()} [/INST]"""

def send_request(url):
    global input_text
    global all_len
    global flag
    global len_max
    global len_min
    post_dict = {"prompt": input_text, "stream": False, "max_tokens": 512, "top_p": 0.9, "frequency_penalty": 0, "use_beam_search": False}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json.dumps(post_dict), headers=headers)
    print(response.text)
    a = eval(response.text)
    if not flag:
        print(a['response'])
        flag = 1
    with open('./response.txt','a', encoding='utf-8') as file:
        file.write(a['response'][0])
        file.write('\n')
        file.write('******************************************\n')
    
    len_max = max(len(a['response'][0].split(' ')), len_max)
    len_min = min(len(a['response'][0].split(' ')), len_min)
    all_len += len(a['response'][0].split(' '))
    
    return response.status_code
url = 'http://37.224.68.132:27194/generate'
# url = 'http://37.224.68.132:27195/generate'
urls = []
for i in range(concurrency):
    urls.append(url)
with ThreadPoolExecutor(max_workers=concurrency) as executor:
    results = executor.map(send_request, urls)

success_count = 0
total_count = 0

for result in results:
    total_count += 1
    if result == 200:
        success_count += 1
        # print(f"Response : {response}")
all_time = float(time.time() - start_time)
print(f"total count : {total_count}")
success_rate = success_count / total_count * 100
print(f"Success rate : {success_rate}%")
print(f"Average time : {all_time / concurrency}")
print(f"All time : {all_time}")
print(f"Avg len : {all_len / concurrency}")
print(f"Max len : {len_max}")
print(f"Min len : {len_min}")