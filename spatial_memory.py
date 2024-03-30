from utils_toy_simulacra import * 

class SpatialMemoryTree: 
    def __init__(self, f_saved): 
        self.tree = {}
        self.tree = json.load(open(f_saved))
    
    def print_tree(self): 
        def _print_tree(tree, depth):
            dash = " >" * depth
            if type(tree) == type(list()): 
                if tree:
                  print (dash, tree)
                return 
            
            for key, val in tree.items(): 
                if key: 
                  print (dash, key)
                _print_tree(val, depth+1)
        _print_tree(self.tree, 0)
    
    def get_str_accessible_sectors(self, curr_world):
        return ", ".join(list(self.tree[curr_world].keys()))
    
    def get_str_accessible_sector_arenas(self, sector):
        curr_world, curr_sector = sector.split(":")
        if not curr_sector:
            return ""
        return ", ".join(list(self.tree[curr_world][curr_sector].keys()))

    def get_str_accessible_arena_game_objects(self, arena):
        curr_world, curr_sector, curr_arena = arena.split(":")
        if not curr_arena:
            return ""

        try: 
            x = ", ".join(self.tree[curr_world][curr_sector][curr_arena])
        except:
            x = ", ".join(self.tree[curr_world][curr_sector][curr_arena.lower()])

        return x      