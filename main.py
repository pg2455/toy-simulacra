from utils_toy_simulacra import *
from persona import Persona
from maze import Maze


# Clear file contents
open(SIM_LOGFILE, 'w').close()
open(PROMPT_LOGFILE, 'w').close() 
open(FAILSAFE_LOGFILE, 'w').close()
open(CONVERSATION_LOGFILE, 'w').close() 
open(SCHEDULES_LOGFILE, 'w').close() 
CALL_LOGS = {'api_calls': 0, 'fail_safe_counts': {}}

maze = Maze("the Ville")
curr_time = sim_start_time = datetime(2024, 2, 13, 0, 0, 0) # Start at midnight
seconds_per_step = 10 * 60 # 10 minutes
n_steps = 180

personas = []
for persona_folder in PERSONAS_FOLDER.iterdir():
    personas.append(Persona(persona_folder.name, persona_folder, curr_time, initiate_plan=True))

step = 0
personas = personas[:3]
movements = {}
while step < n_steps:
    
    # update and execute activities
    for persona in personas:
        curr_tile = persona.get_curr_tile()
        new_tile = persona.advance_one_step(maze, personas, curr_time)
        movements[persona.name] = new_tile

    # update location
    for persona in personas:
        new_tile = movements[persona.name]
        if new_tile:            
            maze.remove_subject_events_from_tile(persona.name, curr_tile)
            maze.add_event_from_tile(persona.scratch.get_curr_event_and_desc(), new_tile)
            persona.move(new_tile)
        
        tile_path = maze.get_tile_path(persona.scratch.curr_tile, level='object')
        string = f"{curr_time.strftime('%H:%M')} {persona.name} {persona.scratch.curr_tile} {persona.scratch.act_description} {persona.scratch.act_address} {tile_path}"
        print_to_file(string, SIM_LOGFILE)
    
    step += 1
    curr_time = sim_start_time + timedelta(seconds=seconds_per_step*step)
    print(f"Current time: {curr_time.strftime('%H:%M')}")