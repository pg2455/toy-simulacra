from utils_toy_simulacra import *
from scratch import Scratch
from associative_memory import AssociativeMemory
from spatial_memory import SpatialMemoryTree
from generate_interaction import *
from generate_schedule import *
from cognition import *

class Persona:
    def __init__(self, name, folder_mem, curr_time, initiate_plan=True):
        self.name = name
        self.s_mem = SpatialMemoryTree(f"{folder_mem}/bootstrap_memory/spatial_memory.json")
        self.a_mem = AssociativeMemory(f"{folder_mem}/bootstrap_memory/associative_memory")
        self.scratch = Scratch(f"{folder_mem}/bootstrap_memory/scratch.json")
        self.scratch.curr_time = curr_time
        
        if initiate_plan:
            self.generate_day_plan(first_day=True)

    def generate_day_plan(self, first_day):
        # To ensure continuity in plans, we update currently that is akin to reflection on the day and broad goals for the next day
        if not first_day:
            self.scratch.currently = get_new_currently(self)
            
        # Generating persona's daily plan in the short term memory
        self.scratch.daily_req = generate_day_plan(self)
        self.scratch.f_daily_schedule = generate_hourly_schedule(self) # Breaks down the plan into sub units with coherence across the time
        self.scratch.f_daily_schedule_hourly_org[:] = (self.scratch.f_daily_schedule)

        # Adding the broad plan to the long term memory (associative memory)
        curr_date = self.scratch.curr_time.strftime("%A %B %d")
        thought = f"This is {self.scratch.first_name}'s plan for {curr_date}"
        for i in self.scratch.daily_req:
            thought += f" {i},"
        thought = thought[:-1] + "."
        created = self.scratch.curr_time
        expiration =  self.scratch.curr_time + timedelta(days=7) ## EXPIRY = 7 days
        s, p, o = (self.scratch.name, "plan", curr_date)
        keywords = set(["plan"])
        poignancy = 5
        thought_embedding_pair = (thought, get_embedding(thought))
        self.a_mem.add_node("thought", created, expiration, s, p, o,
                               thought, keywords, poignancy, thought_embedding_pair, None)

    def perceive_and_retrieve_and_focus(self, maze):
        if self.scratch.act_description is not None and "sleeping" in self.scratch.act_description:
            return
        perceived = perceive(self, maze) # Adds new information to the memory.
        retrieved = retrieve(self, perceived)
        # Retrieve relevant information from the memory and choose to focus on it, if relevant.
        if not retrieved.keys():
            return
        focus_event = choose_retrieved(self, retrieved)
        return focus_event

    def advance_one_step(self, maze, personas, curr_time):
        # Obeserve the surroundings, adjust the memory, retrieve relevant information from the memory, chose the event to react to
        focus_event = self.perceive_and_retrieve_and_focus(maze)

        if focus_event:
            react_mode, other = should_react(self, focus_event, personas)
            if react_mode == 2 and self.scratch.act_event[1] != "chat with": 
                # If not chatting already, Open a conversation and adjust the schedule
                self.open_conversation(maze, other)
                
        if self.scratch.curr_time.strftime('%A %B %d') != curr_time.strftime('%A %B %d'):
            self.generate_day_plan(first_day=False)
            
        self.scratch.curr_time = curr_time
        if self.scratch.act_check_finished():
            new_action = determine_action(self, maze) # NOTE: this function call makes changes to f_daily_schedule
            self.scratch.add_new_action(**new_action)

            # determine it's location [don't change anything yet. Changing it here will lead to other personas not observing this event if reuqired]
            action_address = new_action['action_address']
            target_tiles = maze.address_tiles[action_address]
            new_tile = random.sample(list(target_tiles), 1)[0]
            return new_tile
        return None

    def open_conversation(self, maze, other):
        convo, convo_duration_min = generate_convo(maze, self, other)
        convo_summary = generate_convo_summary(self, other, convo)
        print_convo(convo, convo_duration_min, convo_summary, self.scratch.curr_time, self)

        for person, other_person in [(self, other), (other, self)]:
            # new actions for each of self and other
    
            act_address = f"<persona> {other_person.name}"
            act_event = (person.name, "chat with", other_person.name)
            chatting_with = other.name
            chatting_with_buffer = {other.name: 800}
            chatting_end_time = self.scratch.curr_time + timedelta(minutes=convo_duration_min)
            chatting_end_time += timedelta(seconds=60 - chatting_end_time.second)
            
            new_action = {
                "action_address": act_address,
                "action_duration": int(convo_duration_min),
                "action_description": convo_summary,
                "action_event": act_event,
                "chatting_with": chatting_with,
                "chat": convo,
                "chatting_with_buffer": chatting_with_buffer,
                "chatting_end_time":chatting_end_time,
                "act_obj_description": None, 
                "act_obj_event": None
            }
            person.scratch.add_new_action(**new_action)
            # NOTE: In this tutorial, we are not letting conversation affect the new schedules.
            # curr_index = person.scratch.get_f_daily_schedule_index(main=True)
            # new_schedule, start_index, end_index = generate_updated_schedule(person, convo_summary, convo_duration_min)
            # person.scratch.f_daily_schedule[start_index: end_index] = new_schedule

    def get_curr_tile(self):
        return self.scratch.curr_tile

    def move(self, new_tile):
        self.scratch.curr_tile = new_tile
        