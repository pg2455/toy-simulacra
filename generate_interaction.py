from utils_toy_simulacra import *

def generate_poignancy_score(persona, event_type, description):
    if "is idle" in description:
        return 1

    prompt_template_file = str(TEMPLATE_FOLDER / f"generate_{event_type}_poignancy_score.txt")
        
    prompt_inputs = [
        persona.name,
        persona.scratch.get_str_iss(),
        description,
    ]

    prompt = generate_prompt(prompt_inputs, prompt_template_file)
    score = safe_prompting(prompt, GPT_PARAMS, lambda x:x)
    
    print_prompt(f"generate_poignancy_score -- {event_type}", persona, prompt, score, GPT_PARAMS)

    try: return int(score)
    except:
        string = ">"*50 + "<"*50 + "\n" + f"generate_poignancy_score -- {event_type} @ {persona.name} @ {description} --- Response: {score}\n"
        string += f"response: {score}\n"
        string += f"failed: int(score)\n"
        string += "default: 0\n\n"
        print_failsafe("generate_poignancy_score", string)

        return 0
  

def generate_reaction_type(persona, focused_event, other_persona):
    prompt_template_file = str(TEMPLATE_FOLDER / "generate_reaction_type.txt")
    curr_time_str = persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    focused_event_description = focused_event['curr_event'].description

    name = persona.name
    # relevant events from persona's memory
    context_str = f"{name} just observed {focused_event_description}."
    context_str += f"These are the past relevant events in {name}'s experience:\n"
    for node in focused_event['events']:
        context_str += f"{node.description}. "
    
    # relevant thoughts from persona's memory
    context_str += f"\nThese are the relevant thoughts in {name}'s mind:"
    for node in focused_event['thoughts']:
        context_str += f"{node.description}. "

    # what is persona doing right now
    persona_action_desc = persona.scratch.act_description

    # what is other_persona doing right now
    other_persona_action_desc = other_persona.scratch.act_description
    
    prompt_inputs = [
        context_str,
        curr_time_str,
        persona_action_desc,
        other_persona_action_desc,
        name,
        other_persona.scratch.name,
        persona.scratch.get_str_iss(),
        other_persona.scratch.get_str_iss(),
    ]

    prompt = generate_prompt(prompt_inputs, prompt_template_file)
    response = safe_prompting(prompt, GPT_PARAMS, lambda x:x)

    print_prompt(f"generate_reaction_type", persona, prompt, response, GPT_PARAMS)
    
    try:
        return int(response.split(":")[0].split("Option ")[-1])
    except:
        string = ">"*50 + "<"*50 + "\n" + f"generate_reaction_type -- {persona.scratch.name} -- {other_persona.scratch.name} -- {curr_time_str}\n"
        string += f"response: {response}\n"
        string += "failed: int(response.split(':')[0].split('Option ')[-1])\n"
        string += "default: Option 2\n\n"
        print_failsafe("generate_reaction_type", string)
        return 2


def generate_convo(maze, persona, other_persona):
    convo = simulate_convo(maze, persona, other_persona)

    all_utt = ""
    for row in convo:
        speaker = row[0]
        utt = row[1]
        all_utt += f"{speaker}: {utt}\n"

    # Heuristic: 30 words per minute where each word has 8 characters on average
    # Note: usual statistics is different; 120 words per minute in normal conversation. 
    convo_duration_min = math.ceil(int(len(all_utt)/8 / 30))
    return convo, convo_duration_min


def simulate_convo(maze, persona, other_persona):
    curr_chat = []
    for i in range(8):
        for speaker, listener in [(persona, other_persona), (other_persona, persona)]:
            focal_points = [f"{listener.scratch.name}"]
            retrieved_nodes = extract_relevant_nodes(speaker, focal_points, count=50) # What does agent know about the other persona?
            relationship = generate_summarize_relationship(speaker, listener, retrieved_nodes) # summarize relationship between them
    
            # Create new focal points for the new conversation
            focal_points = [
                f"{relationship}",
                f"{listener.scratch.name} is {listener.scratch.act_description}"
            ]
            last_chat = [": ".join(i) + "\n" for i in curr_chat[-4:]]
            if last_chat:
                focal_points.append("".join(last_chat))
            retrieved_nodes = extract_relevant_nodes(speaker, focal_points, count=15)
            utterance, end = generate_one_utterance(maze, speaker, listener, retrieved_nodes, curr_chat)
            curr_chat += [[speaker.scratch.name, utterance.strip()]]

            if end:
                break
    
        if end: 
            break
    
    return curr_chat


def generate_summarize_relationship(persona, other_persona, retrieved_nodes):
    prompt_template_file = str(TEMPLATE_FOLDER / "summarize_relationship.txt")
    all_embedding_keys = list()
    all_embedding_keys = [f"{i.embedding_key}\n" for key, val in retrieved_nodes.items() for i in val]
    all_embedding_keys_str = "".join(all_embedding_keys)
    prompt_inputs = [
        all_embedding_keys_str,
        persona.scratch.name,
        other_persona.scratch.name
    ]

    prompt = generate_prompt(prompt_inputs, prompt_template_file)
    response = safe_prompting(prompt, GPT_PARAMS, lambda x:x)

    print_prompt(f"generate_summarize_relationship", persona, prompt, response, GPT_PARAMS)
    return response


def generate_one_utterance(maze, speaker, listener, retrieved_nodes, curr_chat):
    prompt_template_file = str(TEMPLATE_FOLDER / "generate_one_utterance.txt")
    curr_context = f"""
    {speaker.scratch.name} was {speaker.scratch.act_description} when {speaker.scratch.name} saw {listener.scratch.name}
    in the middle of {listener.scratch.act_description}.\n{speaker.scratch.name} is initiating a conversation with {listener.scratch.name}.
    """
    retrieved_memory_str = [f"- {v.description}\n" for key, vals in retrieved_nodes.items() for v in vals]
    # Adding the last conversation between the two
    prev_convo = ""
    for i in speaker.a_mem.seq_chat:
        if i.object == listener.scratch.name:
            mins_ago = int((speaker.scratch.curr_time - i.created).total_seconds()/60)
            prev_convo = f"{str(mins_ago)} minutes ago, {speaker.scratch.name} and {listener.scratch.name} were already {i.description}. This context takes place after that conversation."
            break

    tile_info = maze.access_tile(speaker.scratch.curr_tile)
    curr_location = f"{tile_info['arena']} in {tile_info['sector']}"
    if len(curr_chat) == 0:
        convo_str = "[The conversation has not started yet -- start it!]"
    else:
        convo_str = [": ".join(i) + "\n" for i in curr_chat]
        
    prompt_inputs = [
        speaker.scratch.get_str_iss(),
        speaker.scratch.name,
        "".join(retrieved_memory_str),
        prev_convo,
        curr_location,
        curr_context,
        listener.scratch.name,
        "".join(convo_str),
    ]
    prompt = generate_prompt(prompt_inputs, prompt_template_file)
    response = safe_prompting(prompt, GPT_PARAMS, lambda x:x)
    response = response.strip()
    print_prompt(f"generate_one_utterance -- speaker: {speaker.scratch.name} -- listener: {listener.name}", speaker, prompt, response, GPT_PARAMS)

    try:
        x = response.strip().split("\n")
        utt = x[0].split(f"{speaker.scratch.name}:")[-1].strip()
        if len(x) == 2 and x[1] != "" :
            end = False if "False" in x[1] else True
        else:
            end = True
    except Exception as e:
        string = ">"*50 + "<"*50 + "\n" + f"generate_one_utterance -- speaker: {speaker.scratch.name} -- listener: {listener.name}. returning dict()\n"
        string += f"response: {response}\n"
        string += f"failed: response.split(f'{speaker.scratch.name}:')[-1] -- {e}\n"
        string += "default: utt:'' end:True \n\n"
        print_failsafe("generate_one_utterance", string)
        utt, end = "", True

    return utt, end


def generate_convo_summary(persona, other_persona, convo):
    prompt_template_file = str(TEMPLATE_FOLDER / "summarize_convo.txt")
    convo_str  = [": ".join(row) + "\n" for row in convo]
    prompt_inputs = [
        "".join(convo_str),
        persona.scratch.name,
        other_persona.scratch.name,
    ]

    prompt = generate_prompt(prompt_inputs, prompt_template_file)
    response = safe_prompting(prompt, GPT_PARAMS, lambda x:x)

    print_prompt(f"generate_convo_summary -- intiator: {persona.scratch.name} -- other: {other_persona.name}", persona, prompt, response, GPT_PARAMS)

    return response

def generate_updated_schedule(persona, new_activity, new_activity_duration):
    prompt_template_file = str(TEMPLATE_FOLDER / "update_schedule.txt")
    curr_time = persona.scratch.curr_time
    
    # Collect the activities before the current activity and just before the original hour ends
    # And collect the activity which is being truncated
    curr_index_hourly = persona.scratch.get_f_daily_schedule_index(main=False)
    
    total_mins_passed = 0
    for activity, duration in persona.scratch.f_daily_schedule_hourly_org[:curr_index_hourly]:
        total_mins_passed += duration
    start_hour = int(total_mins_passed/60)

    end_hour = start_hour + 1 # default
    end_mins = total_mins_passed
    for activity, duration in persona.scratch.f_daily_schedule_hourly_org[curr_index_hourly:]:
        end_mins += duration 
        if end_mins > curr_time.hour * 60 + curr_time.minute + new_activity_duration:
            end_hour = int(end_mins/60) # inserted activity finishes in time before the next hourly activity
            break        

    curr_hour_activities = []
    truncated_act_dur = []
    duration_sum, count, truncation_finished = 0, 0, False
    count, start_index, end_index = 0, None, None # determines where to insert the new schedule
    for activity, duration in persona.scratch.f_daily_schedule:
        # We are interested in the current hour only
        if (duration_sum >= start_hour * 60) and (duration_sum < end_hour * 60):
            curr_hour_activities.append([activity, duration])
            if start_index is None: start_index = count
            # we record activities up to (and including) the truncated activity here 
            if duration_sum <= total_mins_passed:
                truncated_act_dur.append([activity, duration])
            elif duration_sum > total_mins_passed and not truncation_finished:
                truncated_act_dur.append([activity, duration_sum - total_mins_passed])
                truncation_finished = True

        if duration_sum >= end_hour * 60 and end_index is None:
            end_index = count
        
        duration_sum += duration
        count += 1
        
    # Add the new activity (e.g., conversation) and its duration
    truncated_act_dur.append([new_activity, new_activity_duration])

    # Prepare inputs to the prompt
    start_time_hour = datetime(curr_time.year, curr_time.month, curr_time.day, start_hour)
    end_time_hour = datetime(curr_time.year, curr_time.month, curr_time.day, end_hour)
    
    original_plan = ""
    from_time = start_time_hour
    for activity, duration in curr_hour_activities:
        end_time = from_time + timedelta(minutes=duration)
        original_plan += f"{from_time.strftime('%H:%M')} ~ {end_time.strftime('%H:%M')} -- {activity}\n"
        from_time = end_time

    # Create incomplete string to be completed by the LLM
    new_plan = ""
    from_time = start_time_hour
    for activity, duration in truncated_act_dur:
        end_time = from_time + timedelta(minutes=duration)
        new_plan += f"{from_time.strftime('%H:%M')} ~ {end_time.strftime('%H:%M')} -- {activity}\n"
        from_time = end_time
    new_plan += f"{from_time.strftime('%H:%M')} ~ "
    
    prompt_inputs = [
        persona.scratch.name,
        start_time_hour.strftime('%H:%M'),
        end_time_hour.strftime('%H:%M'),
        original_plan,
        new_activity,
        str(new_activity_duration),
        new_plan,
    ]

    prompt = generate_prompt(prompt_inputs, prompt_template_file)
    response = safe_prompting(prompt, GPT_PARAMS, lambda x:x)

    print_prompt(f"generate_updated_schedule", persona, prompt, response, GPT_PARAMS)

    new_schedule = (prompt + response).split("The revised schedule:")[-1].strip().split("\n")
    new_schedule = [x.split("--") for x in new_schedule]
    new_schedule = [x for x in new_schedule if len(x) == 2]
    updated_schedule = []
    for time_str, activity in new_schedule:
        if "~" not in time_str:
            break # probably the end of the schedule.
        start_time, end_time = time_str.split(" ~ ")
        start_time = datetime.strptime(start_time.strip(), "%H:%M")
        end_time = datetime.strptime(end_time.strip(), "%H:%M")
        delta_min = max(int((end_time - start_time).total_seconds()/60), 0) # non-negative value only
        updated_schedule.append([activity, delta_min])

    string = f"Updated (reason: conversation) -- {persona.name} -- {persona.scratch.curr_time.strftime('%A %B %d %H:%M')}\n"
    string += f"Conversation: {new_activity}\nConversation Duration: {new_activity_duration}"
    print_schedule(string, persona.scratch.f_daily_schedule, persona.scratch.curr_time)

    return updated_schedule, start_index, end_index
    