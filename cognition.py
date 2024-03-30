from utils_toy_simulacra import *
from generate_interaction import generate_reaction_type, generate_poignancy_score 

def perceive(persona, maze):
    curr_tile = persona.scratch.curr_tile
    curr_arena_address = maze.get_tile_path(curr_tile, level='arena')
    nearby_tiles = maze.get_nearby_tiles(persona.scratch.curr_tile, persona.scratch.vision_r)

    percept_events_set = set()
    percept_events_list = []

    # We add new objects to the spatial memory of the agent
    # We also take notice of new events and their distance from the agent
    for i in nearby_tiles:
        tile_info = maze.access_tile(i)

        world, sector = tile_info.get('world', None), tile_info.get('sector', None)
        arena, object = tile_info.get('arena', None), tile_info.get('object', None)
        
        # Initialize spatial memory of persona with this tile
        if world and world not in persona.s_mem.tree: 
            persona.s_mem.tree[world] = {}

        if sector and sector not in persona.s_mem.tree[world]: 
            persona.s_mem.tree[world][sector] = {}

        if arena and arena not in persona.s_mem.tree[world][sector]: 
            persona.s_mem.tree[world][sector][arena] = []

        if object and object not in persona.s_mem.tree[world][sector][arena]: 
            persona.s_mem.tree[world][sector][arena].append(object)

        tile_arena_address = maze.get_tile_path(i, level="arena")
        if tile_info['events'] and tile_arena_address == curr_arena_address:
            dist = math.dist(i, curr_tile)

            # add events
            for event in tile_info['events']:
                if event not in percept_events_set:
                    percept_events_list += [[dist, event]]
                    percept_events_set.add(event)

    # Agent only retains events based on their distance
    percept_events_list = sorted(percept_events_list, key=itemgetter(0))
    perceived_events = [event for dist, event in percept_events_list[:persona.scratch.att_bandwidth]]

    # Add these events (if new) to appropriate memory structure of the agent (associative memory and scratch)
    latest_events = persona.a_mem.get_summarized_latest_events(persona.scratch.retention)
    ret_events = []
    # print(f"LATEST EVENTS: {latest_events}\n\nPERCEIVED EVENTS:{perceived_events}")
    for p_event in perceived_events:
        subject, predicate, object, desc = p_event
        if not predicate:
            predicate, object, desc = "is", "idle", "idle"
        desc = f"{subject.split(':')[-1]} is {desc}"
        p_event = (subject, predicate, object)
        
        if p_event not in latest_events:
            
            sub = p_event[0] if ":" not in p_event[0] else p_event[0].split(":")[-1]
            obj = p_event[2] if ":" not in p_event[2] else p_event[2].split(":")[-1]
            keywords = set([sub, obj])

            # embeddings to represent these events (think of neural encodings in brain, maybe)
            desc_embedding_in = desc
            if "(" in desc:
                # "(xyz)" --> ("xyz")
                desc_embedding_in = (desc_embedding_in.split("(")[1].split(")")[0].strip())

            if desc_embedding_in in persona.a_mem.embeddings:
                event_embedding = persona.a_mem.embeddings[desc_embedding_in]
            else:
                event_embedding = get_embedding(desc_embedding_in)
            event_embedding_pair = (desc_embedding_in, event_embedding)

            # poignancy
            event_poignancy = generate_poignancy_score(persona, "event", desc_embedding_in)

            # Add chats to the memory
            created, expiration = persona.scratch.curr_time, None
            chat_node_ids = []
            if p_event[0] == persona.name and p_event[1] == "chat with":
                curr_event = persona.scratch.act_event
                act_desc = persona.scratch.act_description
                if act_desc in persona.a_mem.embeddings:
                    chat_embedding = persona.a_mem.embeddings[act_desc]
                else:
                    chat_embedding = get_embedding(act_desc)
                chat_embedding_pair = (act_desc, chat_embedding)
                chat_poignancy = generate_poignancy_score(persona, "chat", act_desc)
                chat_node = persona.a_mem.add_node("chat", created, expiration, 
                                                   curr_event[0], curr_event[1], curr_event[2],
                                                   act_desc, keywords, chat_poignancy, chat_embedding_pair,
                                                   persona.scratch.chat)
                chat_node_ids = [chat_node.node_id]            
            
            new_node_in_mem = persona.a_mem.add_node('event', created, expiration, subject, predicate,
                 object, desc, keywords, event_poignancy, event_embedding_pair, chat_node_ids) 
            ret_events.append(new_node_in_mem)
            # persona.scratch.importance_trigger_curr -= event_poignancy # Used in reflection which we do not consider in this tutorial
            # persona.scratch.importance_ele_n += 1

    return ret_events


def retrieve(persona, perceived): 
  # We rerieve events and thoughts separately. 
  retrieved = dict()
  for node in perceived: 
    retrieved[node.description] = dict()
    retrieved[node.description]["curr_event"] = node
    
    relevant_events = persona.a_mem.retrieve_relevant_events(
                        node.subject, node.predicate, node.object)
    retrieved[node.description]["events"] = list(relevant_events)

    relevant_thoughts = persona.a_mem.retrieve_relevant_thoughts(
                          node.subject, node.predicate, node.object)
    retrieved[node.description]["thoughts"] = list(relevant_thoughts)
    
  return retrieved


def choose_retrieved(persona, retrieved):
    """Chooses one of the retrieved events. Currently, it only retrieves if there are other agents around. """
    relevant = []
    # conditions are as used in the simulacra code. See _choose_retrieved in plan.py
    for event_desc, info in retrieved.items():
        node = info['curr_event']
        if node.subject == persona.name: 
            continue
        if (":" not in node.subject and "is idle" not in event_desc):
            relevant.append(info)

    if relevant:
        # print(f"RELEVANT: {relevant}")
        return random.choice(relevant)
    return None

def should_react(persona, focused_event, all_personas):
    """If there should be a reaction. If so, what type of reaction? 0=do nothing, 1=continue with their work (~0), 2=chat."""
    
    if "<waiting>" in persona.scratch.act_address:
        return 0, None
    
    event_node = focused_event['curr_event']

    # In this notebook, we don't converse with other agents --- see the next notebook for that.
    if ":" in event_node.subject: # this is an object --- we don't consider any reaction to the objects in this simulation
        return 0, None

    other_persona = [p for p in all_personas if p.name == event_node.subject]
    if not other_persona:
        return 0, None

    react_mode = generate_reaction_type(persona, focused_event, other_persona[0])
    return react_mode, other_persona[0]