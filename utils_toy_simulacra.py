import json
import os
import time
import random
import re
import math
import numpy as np
from pathlib import Path
from operator import itemgetter
from openai import AzureOpenAI
from datetime import datetime, timedelta

from maze import Maze

CLIENT = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2023-12-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

AZURE_MODEL_MAP = {
  'gpt-3.5-turbo-instruct': 'gpt-35-turbo-instruct',
  'gpt-3.5-turbo': 'gpt-35-turbo'
}

GPT_PARAMS = {
    "engine": "gpt-3.5-turbo-instruct", 
    "max_tokens": 500, 
    "temperature": 1.0, 
    "top_p": 1, 
    "stream": False,
    "frequency_penalty": 0, 
    "presence_penalty": 0, 
    "stop": None
}

HOUR_STR = ["00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", 
              "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", 
              "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", 
              "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
              "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]

TIME_SLEEP_BETWEEN_REQUESTS = 0.1 # seconds

TEMPLATE_FOLDER = Path('./prompt_templates').resolve()
BASE_SIM_FOLDER = Path("./generative_agents/environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/").resolve()
PERSONAS_FOLDER = BASE_SIM_FOLDER / "personas"

PROMPT_LOGFILE = "./prompts_log.txt"
SIM_LOGFILE = "./sim_logs.txt"
CONVERSATION_LOGFILE = "./convo_logs.txt"
FAILSAFE_LOGFILE = "./failsafe_logs.txt"
SCHEDULES_LOGFILE = "./schedules_logfile.txt"
PRINT_SCHEDULE = True
PRINT_PROMPTS = True
PRINT_CONVO = True
PRINT_FAILSAFE = True

CALL_LOGS = {
    "api_calls": 0,
    "fail_safe_counts": {}
}


def print_prompt(fn_name, persona, prompt, response, params, do_not_print=False):
    if not PRINT_PROMPTS:
        return
    
    if do_not_print:
        return
        
    curr_time = persona.scratch.curr_time.strftime('%A %B %d %H:%M:%S')
    with open(PROMPT_LOGFILE, mode="a") as f:
        string = "\n\n" + ">"*50 + "<"*50 + "\n\n" + str(params) + "\n\n"
        print(f"{string}{curr_time}\n{fn_name} --- {persona.name}\n\n --- PROMPT: ---\n{prompt}\n\n--- RESPONSE: ---\n{response}", file=f)

def normalize(seq):
    _min = min(seq)
    _max = max(seq)
    return [(i - _min) / (1e-6 + _max-_min) for i in seq]

def cos_sim(a,b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def print_to_file(string, logfile):
    with open(logfile, 'a') as f:
        print(string, file=f)

def print_convo(convo, convo_duration_min, convo_summary, curr_time, persona):
    if not PRINT_CONVO:
        return
    string = f"{curr_time.strftime('%A %B %d %H:%M')} -- Initiator: {persona.name}\n\n"
    string += f"Summary: {convo_summary}\n"
    string += f"Time taken (minutes): {str(convo_duration_min)}\n"
    string += "".join([": ".join(i) + "\n" for i in convo])
    with open(CONVERSATION_LOGFILE, 'a') as f:
        print(string, file=f)

def print_failsafe(fn_name, string):
    if not PRINT_FAILSAFE:
        return 

    if fn_name in CALL_LOGS['fail_safe_counts']:
        CALL_LOGS['fail_safe_counts'][fn_name] += 1
    else:
        CALL_LOGS['fail_safe_counts'][fn_name] = 0

    string = f"Fn: {string}"
    with open(FAILSAFE_LOGFILE, 'a') as f:
        print(string, file=f)
    
def print_schedule(string, schedule, curr_time):
    if not PRINT_SCHEDULE:
        return

    string = f"{string}\n"
    t = midnight = datetime(curr_time.year, curr_time.month, curr_time.day)
    for activity, duration in schedule:
        t += timedelta(minutes=duration)
        string += f"{t.strftime('%H:%M')} -- {activity}\n"

    with open(SCHEDULES_LOGFILE, "a") as f:
        print(string, file=f)

# Basic prompting helper functions

def get_embedding(text):
    text = text.replace("\n", " ")
    if not text:
        text = "blank"

    response = CLIENT.embeddings.create(
        input = [text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def generate_prompt(prompt_inputs, template_file):
    with open(template_file) as f:
        template = f.read()

    prompt = template.split("<prompt_start>###</prompt_start>")[1]
    for count, input in enumerate(prompt_inputs):
        replace_str = input if input is not None else ""
        prompt = prompt.replace(f"!<INPUT {count}>!", replace_str)

    return prompt.strip()

def prompt_gpt(prompt, parameters):
    try:
        response = CLIENT.completions.create(
            model=AZURE_MODEL_MAP[parameters["engine"]],
            prompt=prompt,
            temperature=parameters["temperature"],
            max_tokens=parameters["max_tokens"],
            top_p=parameters["top_p"],
            frequency_penalty=parameters["frequency_penalty"],
            presence_penalty=parameters["presence_penalty"],
            stream=parameters["stream"],
            stop=parameters["stop"]
        )
        return response.choices[0].text
    except Exception as e:
        print(e)
        return -1


def prompt_gpt4(prompt, parameters):
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people complete the text either by continuing where they leave or by following the instructions. Don't write unnecessary text. Only the asked tasks.",
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    try:
        response = CLIENT.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=parameters["temperature"],
            max_tokens=parameters["max_tokens"],
            top_p=parameters["top_p"],
            frequency_penalty=parameters["frequency_penalty"],
            presence_penalty=parameters["presence_penalty"],
            stop=parameters["stop"]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return -1

def safe_prompting(prompt, parameters, func_clean_up, func_validate=None, repeat=5):
    if func_validate is None:
        def func_validate(response):
            try: func_clean_up(response)
            except: return False
            return True

    for i in range(repeat):
        curr_response = PROMPT_FN(prompt, parameters)
        CALL_LOGS["api_calls"] += 1
        if func_validate(curr_response):
            return func_clean_up(curr_response)
        else:
            time.sleep(TIME_SLEEP_BETWEEN_REQUESTS)

    print(f"{prompt} failed after {repeat} attempt. Returning None.")
    return None

def extract_relevant_nodes(persona, queries, count=30):
    """Retrieves nodes from agent's memory relevant to all `query` in `queries`."""
    nodes = []
    for node in persona.a_mem.seq_thought + persona.a_mem.seq_event + persona.a_mem.seq_chat:
        if "idle" not in node.embedding_key:
            nodes.append([node.last_accessed, node])
    nodes = sorted(nodes, key=lambda x:x[0])
    nodes = [node for _, node in nodes]

    persona_receny_w = persona.scratch.recency_decay
    recency = normalize([persona_receny_w**i for i in range(1, len(nodes) + 1)])
    importance = normalize([node.poignancy for node in nodes])
    
    retrieved = dict()
    v1, v2, v3 = persona.scratch.recency_w, persona.scratch.relevance_w, persona.scratch.importance_w
    w1, w2, w3 = 0.5, 3, 2 ## HARD CODED WEIGHTS 
    for query in queries:
        query_embedding = get_embedding(query)
        node_relevance = normalize([cos_sim(node.embedding, query_embedding) for node in nodes])
        node_relevance = [x*v1*w1 + y*v2*w2 + z*v3*w3 for x,y,z in zip(recency, importance, node_relevance)]
        top_nodes = sorted([(val, idx) for idx, val in enumerate(node_relevance)], key=lambda x:x[0])[-count:]
        for _, idx in top_nodes:
            nodes[idx].last_accessed = persona.scratch.curr_time
        retrieved[query] = [nodes[idx] for _, idx in top_nodes]

    return retrieved



#### DEFINE YOUR PROMPT_FN
PROMPT_FN = prompt_gpt