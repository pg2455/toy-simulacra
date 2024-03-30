from utils_toy_simulacra import *

class Scratch:
    def __init__(self, filename):
        scratch = json.load(open(filename))

        # PERSONA HYPERPARAMETERS
        self.vision_r = scratch.get('vision_r', 4) # Radius of visible boundaries
        self.att_bandwidth = scratch.get('att_bandwidth', 3) # ??
        self.retention = scratch.get('retention', 5) # ??

        # CORE IDENTITY 
        self.name = scratch.get('name', None)
        self.first_name = scratch.get('first_name', None)
        self.last_name = scratch.get('last_name', None)
        self.age = scratch.get('age', None)
        self.innate = scratch.get('innate', None) # Nature of the person
        self.learned = scratch.get('learned', None) # Learned traits of the person
        self.currently = scratch.get('currently', None) # Any current plans?
        self.lifestyle = scratch.get('lifestyle', None) # Lifestyle of the person
        self.living_area = scratch.get('living_area', None) # General living area; area where they spent the time most outside of work

        # REFLECTION VARIABLES
        self.concept_forget = scratch.get('concept_forget', 100)
        self.daily_reflection_time = scratch.get('daily_reflection_time', 60 * 3)
        self.daily_reflection_size = scratch.get('daily_reflection_size', 5)
        self.overlap_reflect_th = scratch.get('overlap_reflect_th', 4)
        self.kw_strg_event_reflect_th = scratch.get('kw_strg_event_reflect_th', 10)
        self.kw_strg_thought_reflect_th = scratch.get('kw_strg_thought_reflect_tg', 4)

        # NEW REFLECTION VARIABLES
        self.recency_w = scratch.get('recency_w', 1)
        self.relevance_w = scratch.get('relevance_w', 1)
        self.importance_w = scratch.get('importance_w', 1)
        self.recency_decay = scratch.get('recency_decay', 0.99)
        self.importance_trigger_max = scratch.get('importance_trigger_max', 150)
        self.importance_trigger_curr = scratch.get('importance_trigger_curr', 150)
        self.important_ele_n = scratch.get('important_ele_n', 0)
        self.thought_count = scratch.get('thought_count', 5)

        # WORLD INFORMATION
        self.curr_time = scratch.get('curr_time', datetime.now()) # What's the current time?
        self.curr_tile = scratch.get('curr_tile', None) # Where is the person now?
        self.daily_plan_req = scratch.get('daily_plan_req', '') # What's a typical daily plan?
        tile_filename = '/'.join(filename.split('/')[:-4]) + '/environment/0.json'
        curr_tile = json.load(open(tile_filename))[self.name]
        self.curr_tile = (curr_tile['x'], curr_tile['y'])
        

        # PLANNING VARIABLES
        self.daily_req = scratch.get('daily_req', [])
        self.f_daily_schedule = scratch.get('f_daily_schedule', []) # This is changed every time the hourly activity is decomposed
        self.f_daily_schedule_hourly_org = scratch.get('f_daily_schedule_hourly_org', []) # This remains same as f_daily_schedule

        # ACTIONS OF THE PERSON
        self.act_address = scratch.get('act_address', None)
        self.act_start_time = scratch.get('act_start_time', None)
        self.act_duration = scratch.get('act_duration', None)
        self.act_description = scratch.get('act_description', None)
        self.act_event = scratch.get('act_event', (self.name, None, None)) # See the section on events.

        self.act_path_set = scratch.get('act_path_set', False)
        self.planned_path = scratch.get('planned_path', [])

        # CONVERSATION VARIABLES
        self.chatting_with = scratch.get('chatting_with', None)
        self.chat = scratch.get('chat', None)
        self.chatting_with_buffer = scratch.get('chatting_with_buffer', dict())
        self.chatting_end_time = scratch.get('chatting_end_time', None)

    def get_f_daily_schedule_index(self, advance=0, main=True):
        """
        Returns the index of action that is taking place now (advance=0) or sometime in future for a non-zero advance minutes. 
        """ 
        total_time_elapsed = self.curr_time.hour * 60 
        total_time_elapsed += self.curr_time.minute + advance

        ref_list = self.f_daily_schedule if main else self.f_daily_schedule_hourly_org
        elapsed, curr_index = 0, 0
        for task, duration in ref_list:
            elapsed += duration
            if elapsed > total_time_elapsed:
                return curr_index
            curr_index += 1
        return curr_index

    def get_str_iss(self):
        # ISS stands for Identity Stable Set - a bare minimum description of the persona that is used in prompts that need to call on the persona.
        commonset = ""
        commonset += f"Name: {self.name}\n"
        commonset += f"Age: {self.age}\n"
        commonset += f"Innate traits: {self.innate}\n"
        commonset += f"Learned traits: {self.learned}\n"
        commonset += f"Currently: {self.currently}\n"
        commonset += f"Lifestyle: {self.lifestyle}\n"
        commonset += f"Daily plan requirement: {self.daily_plan_req}\n"
        if self.curr_time:
            commonset += f"Current Date: {self.curr_time.strftime('%A %B %d')}\n"
        return commonset

    def add_new_action(self, 
                       action_address,
                       action_duration,
                       action_description,
                       action_event,
                       chatting_with, 
                       chat, 
                       chatting_with_buffer,
                       chatting_end_time,
                       act_obj_description,
                       act_obj_event,
                       act_start_time=None):
        self.act_address = action_address
        self.act_duration = action_duration
        self.act_description = action_description
        self.act_event = action_event

        self.chatting_with = chatting_with
        self.chat = chat
        if chatting_with_buffer:
            self.chatting_with_buffer.update(chatting_with_buffer)

        self.chatting_end_time = chatting_end_time 
        self.act_start_time = self.curr_time # This is the start time of the action
        self.act_path_set = False

    def act_check_finished(self):
        # Returns True if the action has finished
        if not self.act_address:
            return True

        # Compute the end time for the chat
        if self.chatting_with:
            end_time = self.chatting_end_time
        else:
            x = self.act_start_time
            if x.second != 0:
                x = x.replace(second=0)
                x = (x + timedelta(minutes=1))
            end_time = (x + timedelta(minutes=self.act_duration))

        if end_time < self.curr_time:
            return True

        return False

    def act_summarize_str(self):
        start_datetime_str = self.act_start_time.strftime('%A %B %d -- %H:%M %p')
        x = f"[{start_datetime_str}]\n"
        x += f"Activity: {self.name} is {self.act_description}\n"
        x += f"Address: {self.act_address}\n"
        x += f"Duration in minutes (e.g., x min): {str(self.act_duration)} min\n"
        return ret

    def get_curr_event_and_desc(self): 
        if not self.act_address: 
          return (self.name, None, None, None)
        else: 
          return (self.act_event[0], 
                  self.act_event[1], 
                  self.act_event[2],
                  self.act_description)
