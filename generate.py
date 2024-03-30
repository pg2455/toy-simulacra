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
        start_idx = response.find('{')
        end_idx = response.find('}')
        output = json.loads(response[start_idx: end_idx + 1])
    except Exception as e:
        string = ">"*50 + "<"*50 + "\n" + f"generate_one_utterance -- speaker: {speaker.scratch.name} -- listener: {listener.name}. returning dict()\n"
        string += f"response: {response}\n"
        string += f"failed: json.loads(response.strip()) -- {e}\n"
        string += "default: dict() \n\n"
        print_failsafe("generate_one_utterance", string)
        
        output = dict()
    utt = output.get(speaker.scratch.name, "")
    end = output.get(f"Did the conversation end with {speaker.scratch.name}'s utterance?", True)
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


def generate_day_plan(persona):
    """Produces broad agenda for the day. """
    prompt_template_file = str(TEMPLATE_FOLDER / "day_planning.txt")
    prompt_input = [
        persona.scratch.get_str_iss(),
        persona.scratch.lifestyle,
        persona.scratch.curr_time.strftime("%A %B %d"),
        persona.scratch.first_name
    ]
    
    prompt = generate_prompt(prompt_input, prompt_template_file)
    schedule = safe_prompting(prompt, GPT_PARAMS, lambda x:x)

    print_prompt("generate_daily_plan", persona, prompt, schedule, GPT_PARAMS)

    schedule = prompt + schedule
    schedule = schedule[schedule.find("1)") + 2:]
    schedule = re.split(r'\d+\)', schedule)

    string = f"day high-level schedule -- {persona.name} -- {persona.scratch.curr_time.strftime('%A %B %d %H:%M')}\n"
    print_schedule(string, persona.scratch.f_daily_schedule, persona.scratch.curr_time)

    return [i.strip() for i in schedule]

def get_new_currently(persona):
    """Reflects on the day's activity and returns a new `currently` for persona to take on. """
    name = persona.scratch.name 
    curr_day = persona.scratch.curr_time.strftime("%A %B %d")
    queries = [
        f"{name}'s plan for {curr_day}",
        f"Important recent events for {name}'s life."
    ]
    retrieved = extract_relevant_nodes(persona, queries, count=30)

    # Add statements about the retrieved nodes
    statements = "[Statements]\n"
    for query, nodes in retrieved.items():
        for node in nodes:
            statements += f"{node.created.strftime('%A %B %d -- %H:%M %p')}: {node.embedding_key}\n"

    # Create a broad agenda for the next day
    planning_prompt = f"""{statements}
    Given the statements above, is there anything that {name} should remember as they plan for *{curr_day}*?
    If there is any scheduling information, be as specific as possible (including date, time, and location if stated in the statement).\n
    Write the response from {name}'s perspective.
    """
    params = GPT_PARAMS.copy()
    params['model'] = "gpt-3.5-turbo"
    params['max_tokens'] = 1000
    params['temperature'] = 0.8
    plan_note = safe_prompting(planning_prompt, params, lambda x:x)

    print_prompt("get_new_currently --> plan_note", persona, planning_prompt, plan_note, params)

    thought_prompt = f"""{statements}
    Given the statements above, how might we summarize {name}'s feelings about their days up to now?\n
    Write the response from {name}'s perspective.
    """
    thought_note = safe_prompting(thought_prompt, params, lambda x:x)

    print_prompt("get_new_currently --> thought_note", persona, thought_prompt, thought_note, params)

    prev_currently = persona.scratch.currently
    prev_day = persona.scratch.curr_time - timedelta(days=1)
    prev_day = prev_day.strftime('%A %B %d')
    update_currently_prompt = f"""
    {name}'s status from {prev_day}: {prev_currently}\n\n
    {name}'s thoughts at the end of {prev_day}: {plan_note} {thought_note}\n\n
    It is now {curr_day}. Given the above, write {name}'s status for {curr_day} that reflects {name}'s thoughts at the end of {curr_day}.
    Write this in third-person talking about {name}.
    If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement).\n\n
    Follow this format below:\nStatus: <new_status>
    """
    new_currently = safe_prompting(update_currently_prompt, params, lambda x:x)

    print_prompt("get_new_currently --> new_currently", persona, update_currently_prompt, new_currently, params)

    return new_currently

def extract_relevant_nodes(persona, queries, count=30):
    """Retrieves nodes from agent's memory relevant to all `query` in `queries`."""
    nodes = []
    for node in persona.a_mem.seq_thought + persona.a_mem.seq_event: ## ADD seq_chat here
        if "idle" not in node.embedding_key:
            nodes.append([node.last_accessed, node])
    nodes = sorted(nodes, key=lambda x:x[0])
    nodes = [node for _, node in nodes]

    persona_receny_w = persona.scratch.recency_decay
    recency = _normalize([persona_receny_w**i for i in range(1, len(nodes) + 1)])
    importance = _normalize([node.poignancy for node in nodes])
    
    retrieved = dict()
    v1, v2, v3 = persona.scratch.recency_w, persona.scratch.relevance_w, persona.scratch.importance_w
    w1, w2, w3 = 0.5, 3, 2 ## HARD CODED WEIGHTS 
    for query in queries:
        query_embedding = get_embedding(query)
        node_relevance = _normalize([cos_sim(node.embedding, query_embedding) for node in nodes])
        node_relevance = [x*v1*w1 + y*v2*w2 + z*v3*w3 for x,y,z in zip(recency, importance, node_relevance)]
        top_nodes = sorted([(val, idx) for idx, val in enumerate(node_relevance)], key=lambda x:x[0])[-count:]
        for _, idx in top_nodes:
            nodes[idx].last_accessed = persona.scratch.curr_time
        retrieved[query] = nodes

    return retrieved

def generate_hourly_schedule(persona):
    """Uses broad agenda for the day to plan hourly schedule."""
    curr_date = persona.scratch.curr_time.strftime("%A %B %d")
    prompt_template_file = str(TEMPLATE_FOLDER / "hourly_planning.txt")
    # Example of a schedule
    schedule_format = ""
    for hour in HOUR_STR:
        schedule_format += f"[{curr_date} -- {hour}]"
        schedule_format += f" Activity: [Fill in]\n"
    schedule_format = schedule_format[:-1]

    # Broad plan of the persona
    plan_str = f"Here is the orginally intended today's schedule of {persona.scratch.first_name}: "
    for count, activity in enumerate(persona.scratch.daily_req):
        plan_str += f"({str(count+1)}) {activity}, "
    plan_str = plan_str[:-2]
    plan_str += f"\nIf {persona.scratch.first_name} is sleeping, use 'sleeping' as the activity"

    prompt_inputs = [schedule_format, persona.scratch.get_str_iss(), plan_str, None, None]

    # today's prior schedule (needed for coherence)
    activities_list = []
    prior_schedule = "\n"
    for count, hour in enumerate(HOUR_STR):
        # prepare the string for prior schedule
        if count > 0:
            prior_schedule += f"{curr_date} -- {HOUR_STR[count-1]} Acitvity:"
            prior_schedule += f" {persona.scratch.first_name} is {activities_list[count - 1]}\n"
            prompt_inputs[-2] = prior_schedule

        # final prompt to be completed
        final_prompt = f" [{curr_date} -- {hour}] Activity: {persona.scratch.first_name} is"
        prompt_inputs[-1] = final_prompt

        # modify the parameters because we don't need to generate a lot of tokens
        prompt = generate_prompt(prompt_inputs, prompt_template_file)
        params = GPT_PARAMS.copy()
        params['stop'] = ['\n']
        params['temperature'] = 0.5
        params['max_tokens'] = 50
        next_hour_activity = safe_prompting(prompt, params, lambda x:x)

        print_prompt("generate_hourly_schedule", persona, prompt, next_hour_activity, params)
        
        activities_list.append(next_hour_activity.strip())

    # post-processing the output
    compressed_list = [('###', 0)]
    for activity in activities_list:
        if compressed_list[-1][0] == activity:
            compressed_list[-1][1] += 1
        else:
            compressed_list.append([activity, 1])
    compressed_list.pop(0)

    string = f"hourly schedule -- {persona.name} -- {persona.scratch.curr_time.strftime('%A %B %d %H:%M')}\n"
    print_schedule(string, persona.scratch.f_daily_schedule, persona.scratch.curr_time)
    
    return [(x, y*60) for x,y in compressed_list]

def generate_task_decompose(persona):
    """Generates 5 min increments of the current action for duration of that action."""
    curr_date = persona.scratch.curr_time.strftime("%A %B %d")
    prompt_template_file = str(TEMPLATE_FOLDER / "decompose_task.txt")
    
    curr_f_org_index = persona.scratch.get_f_daily_schedule_index(main=False) # gets from f_daily_schedule_hourly_org
    # print(curr_f_org_index, persona.scratch.f_daily_schedule_hourly_org, len(persona.scratch.f_daily_schedule_hourly_org))
    # Prepare a summary string to capture an hour before and an hour after the current action event
    summary_str = f"Today is {curr_date}. From "
    for index in [curr_f_org_index- 1, curr_f_org_index, curr_f_org_index+1]:
        if index >= len(persona.scratch.f_daily_schedule_hourly_org) or index < 0:
            continue

        start_min = sum(i[1] for i in persona.scratch.f_daily_schedule_hourly_org[:index])
        action, time_elapsed = persona.scratch.f_daily_schedule_hourly_org[index]
        start_time = datetime.strptime("00:00:00", "%H:%M:%S") + timedelta(minutes=start_min)
        end_time = start_time + timedelta(minutes=time_elapsed)

        start_time_str, end_time_str = start_time.strftime('%H:%M%p'), end_time.strftime('%H:%M%p')
        summary_str += f"{start_time_str} ~ {end_time_str}, {persona.name} is planning {action}, "

        if index == curr_f_org_index: # We are interested in decomposing the activity at curr_f_org_index
            curr_time_range, curr_time_duration, curr_action_desc = f"{start_time_str} ~ {end_time_str}", str(time_elapsed), action
            total_time_range = time_elapsed

    summary_str = summary_str[:-2] + ". "

    prompt_inputs = [
        persona.scratch.get_str_iss(),
        summary_str,
        persona.scratch.first_name,
        curr_action_desc,
        curr_time_range,
        curr_time_duration
    ]

    params = GPT_PARAMS.copy()
    params['temperature'] = 0.8 ## Empirically, so that it doesn't deviate from the output format.
    prompt  = generate_prompt(prompt_inputs, prompt_template_file)
    response = safe_prompting(prompt, params, lambda x:x)
    
    print_prompt("generate_task_decompose", persona, prompt, response, params)

    full_str = prompt + response
    rem_str = full_str.split("---")[3]
    schedule = re.split(r'\d+\>', rem_str)
    schedule = [i.strip() for i in schedule if i.strip()]

    # post-process this schedule to 5 min increments
    activities = [["dummy", -1]]
    for activity in schedule:
        try:
            task, rest = activity.split("(duration in minutes:")
        except:
            task, rest = activity.split("(duration in minutes") # Failure prevention
            
        if "," not in rest: # FAIL PREVENTION: Sometimes prompt might not give ", minutes left: xx)" in the end as prompted.
            duration = int(rest[:-1])
        else:
            duration = int(rest.split(",")[0])

        activities.append([task.strip(), duration])
    activities = activities[1:]

    # Making sure that the activities fall in the time range.    
    lagging_sum, duration_sum, idx = 0, 0, 0
    output = []
    for task, duration in activities:
        duration_sum += duration
        if duration_sum <= total_time_range:
            output.append([task, duration])
        else:
            output.append([task, total_time_range - lagging_sum])
        idx += 1
        lagging_sum += duration

    return output

def determine_action(persona, maze):
    def determine_decompose(act_desc, act_dura):
        if "sleeping" in act_desc:
            return False
        if act_dura < 60:
            return False 
        return True
    
    curr_action_index = persona.scratch.get_f_daily_schedule_index()

    act_desc, act_dura =  persona.scratch.f_daily_schedule[curr_action_index]
    if determine_decompose(act_desc, act_dura):
        persona.scratch.f_daily_schedule[curr_action_index: curr_action_index+1] = (
            generate_task_decompose(persona) # GPT
        )
        string = f"decomposed -- {persona.name} -- {persona.scratch.curr_time.strftime('%A %B %d %H:%M')}\n"
        string += f"Current action: {act_desc}\nDuration: {act_dura}"
        print_schedule(string, persona.scratch.f_daily_schedule, persona.scratch.curr_time)
        
        # to add up minutes
        total_time_accounted = sum(i[1] for i in persona.scratch.f_daily_schedule)
        if total_time_accounted < 1440:
            persona.scratch.f_daily_schedule += [["sleeping", 1440 - total_time_accounted]]

    act_desc, act_dura = persona.scratch.f_daily_schedule[curr_action_index]

    # Now we determine this action's location to execute
    act_world = maze.access_tile(persona.scratch.curr_tile)['world']
    act_sector = generate_action_sector(act_desc, persona, maze, 
                                         curr_determined_address=act_world) # GPT
    act_arena = generate_action_sector_arena(act_desc, persona, maze, 
                                        curr_determined_address=f"{act_world}:{act_sector}") # GPT
    act_object = generate_action_sector_arena_object(act_desc, persona, maze, 
                                                     curr_determined_address=f"{act_world}:{act_sector}:{act_arena}") # GPT
    new_address = f"{act_world}:{act_sector}:{act_arena}" 
    new_address += f":{act_object}" if act_object else ""
    act_event = generate_action_event_triple(act_desc, persona)
    act_obj_desc = None
    act_obj_event = None

    return {
        "action_address": new_address,
        "action_duration": int(act_dura),
        "action_description": act_desc,
        "action_event": act_event,
        "chatting_with": None, "chat": None, "chatting_with_buffer": None, "chatting_end_time": None,
        "act_obj_description": act_obj_desc,
        "act_obj_event": act_obj_event,
    }
                                   
def generate_action_sector(action, persona, maze, curr_determined_address=None):
    prompt_template_file = str(TEMPLATE_FOLDER / "determine_action_sector.txt")
    curr_tile = persona.scratch.curr_tile
    tile_info = maze.access_tile(curr_tile)
    world, curr_sector = tile_info['world'], tile_info['sector']
    all_sectors = persona.s_mem.get_str_accessible_sectors(f"{world}")
    curr_sector_arenas = persona.s_mem.get_str_accessible_sector_arenas(f"{world}:{curr_sector}")
    
    living_area = persona.scratch.living_area
    living_area_sector = living_area.split(":")[1]
    living_area_arenas = persona.s_mem.get_str_accessible_sector_arenas(f"{world}:{living_area_sector}")
    prompt_inputs = [
        persona.scratch.name,
        living_area_sector,
        living_area_arenas,
        curr_sector,
        curr_sector_arenas,
        all_sectors,
        action
    ]
    
    params = GPT_PARAMS.copy()
    params['max_tokens'] = 20
    params['temperature'] = 0
    params['top_p'] = 1
    prompt  = generate_prompt(prompt_inputs, prompt_template_file)
    response = safe_prompting(prompt, params, lambda x:x)

    print_prompt("generate_action_sector", persona, prompt, response, params)

    return response.split("}")[0]

def generate_action_sector_arena(action, persona, maze, curr_determined_address):
    prompt_template_file = str(TEMPLATE_FOLDER / "determine_action_arena.txt")
    new_sector = curr_determined_address.split(":")[1]
    new_possible_arenas = persona.s_mem.get_str_accessible_sector_arenas(curr_determined_address)
    prompt_inputs = [
        persona.scratch.name,
        new_sector,
        new_possible_arenas,
        action     
    ]
    params = GPT_PARAMS.copy()
    params['max_tokens'] = 20
    params['temperature'] = 0
    params['top_p'] = 1
    prompt  = generate_prompt(prompt_inputs, prompt_template_file)
    response = safe_prompting(prompt, params, lambda x:x)
    
    print_prompt("generate_action_sector", persona, prompt, response, params)

    if response.split("}")[0] not in [x.strip() for x in new_possible_arenas.split(",")]:
        object = random.sample(new_possible_arenas.split(","), 1)[0].strip()
        
        string = ">"*50 + "<"*50 + "\n" + f"new_possible_arenas @ {persona.name} @ {curr_determined_address} @ {action} --> {object}\n"
        string += f"response: {response}\n"
        string += f"failed: response not in {new_possible_arenas.split(',')}\n"
        string += "default: random.sample \n\n"
        print_failsafe("generate_action_sector_arena", string)
    
    return response.split("}")[0]


def generate_action_sector_arena_object(action, persona, maze, curr_determined_address):
    prompt_template_file = str(TEMPLATE_FOLDER / "determine_action_object.txt")
    possible_objects = persona.s_mem.get_str_accessible_arena_game_objects(curr_determined_address)
    prompt_inputs = [
        f"{action}",
        possible_objects    
    ]
    params = GPT_PARAMS.copy()
    params['max_tokens'] = 20
    params['temperature'] = 0.1
    params['top_p'] = 1
    params['stop'] = ["\n"]
    prompt  = generate_prompt(prompt_inputs, prompt_template_file)
    response = safe_prompting(prompt, params, lambda x:x)

    print_prompt("generate_action_sector_arena_object", persona, prompt, response, params)
    
    # Fail safe mechanism
    if response.strip() not in [x.strip() for x in possible_objects.split(",")]:
        object = random.sample(possible_objects.split(","), 1)[0].strip()
        
        string = ">"*50 + "<"*50 + "\n" + f"generate_action_sector_arena_object @ {persona.name} @ {curr_determined_address} @ {action} --> {object}\n"
        string += f"response: {response}\n"
        string += f"failed: response not in {possible_objects.split(',')}\n"
        string += "default: random.sample \n\n"
        print_failsafe("generate_action_sector_arena_object", string)
        
        return object

    return response.strip()

def generate_action_event_triple(action, persona):
    prompt_template_file = str(TEMPLATE_FOLDER / "generate_event_triplet.txt")
    prompt_inputs = [
        persona.scratch.name,
        action.lower()
    ]  
    params = GPT_PARAMS.copy()
    params['max_tokens'] = 50
    params['temperature'] = 0
    params['top_p'] = 1
    prompt  = generate_prompt(prompt_inputs, prompt_template_file)
    response = safe_prompting(prompt, params, lambda x:x)

    print_prompt("generate_action_event_triple", persona, prompt, response, params)

    full_str = prompt + response
    output = full_str.split("---")[-1].split("Output:")[-1].strip()[1:]

    output = [i.strip() for i in output.split(")")[0].split(",")]
    if len(output) != 3:
        output = [persona.scratch.name, 'is', output[-1]]

        string = ">"*50 + "<"*50 + "\n" + f"generate_action_event_triple @ {persona.name} @ {action} --> {output}\n"
        string += f"response: {response}\n"
        string += f"failed: len(output) != 3\n"
        string += "default: output = [persona.scratch.name, 'is', output[-1]] \n\n"
        print_failsafe("generate_action_event_triple", string)
    
    return output