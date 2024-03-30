from utils_toy_simulacra import *

class ConceptNode:
    node_id: int # bookkeeping
    node_count: int #bookkeeping
    node_type: str # thought / event / chat
    type_count: int # bookkeeping
    depth: int # 

    created: int # 
    expiration: int #

    subject: str # subject usually the agent itself
    predicate: str # 
    object: str # object of this event

    description: str # A full description of the event (usually obtained from LLM)
    embedding_key: str # a key to reference while accessing the embeddings of the event
    embedding: int  # a vector instead. 
    poignancy: int # Used for relevance and reflection
    keywords: list # keywords to retrieve this node
    filling: int # ??
    
    def __init__(self, **kwargs):
        for key in self.__annotations__:
            setattr(self, key, kwargs.get(key, None)) 
        self.last_accessed = self.created

    def spo_summary(self):
        return (self.subject, self.predicate, self.object)


class AssociativeMemory:
    def __init__(self, folder_name):
        self.id_to_node = dict()

        self.seq_event = []
        self.seq_thought = []
        self.seq_chat = []

        self.kw_to_event = dict()
        self.kw_to_thought = dict()
        self.kw_to_chat = dict()

        x = json.load(open(folder_name + '/kw_strength.json'))
        self.kw_strength_event = x.get('kw_strength_event', dict())
        self.kw_strength_thought = x.get('kw_strength_thought', dict())

        self.embeddings = json.load(open(folder_name + "/embeddings.json"))
        nodes = json.load(open(folder_name + "/nodes.json"))
        for count in range(len(nodes)):
            node_id = f"node_{str(count+1)}"
            node_details = nodes[node_id]

            node_type = node_details['node_type']
            node_count = node_details['node_count']
            depth = node_details['depth']

            created = datetime.datetime.strptime(node_details['created'], '%Y-%m-%d %H:%M:%S')

            expiration = None
            if node_details['expiration']:
                expiration = datetime.datetime.strptime(node_details['expiration'], '%Y-%m-%d %H:%M:%S')

            subject, predicate, object = node_details['subject'], node_details['predicate'], node_details['object']
            description = node_details['description']
            embedding_pair = (node_details['embedding_key'], self.embeddings[node_details['embedding_key']])
            poignancy = node_details['poignancy']
            keywords = set(node_details['keywords'])
            filling = node_details['filling']

            self.add_node(node_type, created, expiration, subject, predicate,
                 object, description, keywords, poignancy, embedding_pair, filling)
        
    def add_node(self, node_type, created, expiration, subject, predicate,
                 object, description, keywords, poignancy, embedding_pair, filling):

        node_count = len(self.id_to_node.keys()) + 1
        node_id = f'node_{node_count}'

            
        if node_type == 'chat':
            type_count = len(self.seq_chat) + 1
            depth = 0
        elif node_type == 'event':
            type_count = len(self.seq_event) + 1
            depth = 0  
        elif node_type == 'thought':
            type_count = len(self.seq_thought) + 1
            depth = 1
            if filling:
                depth += max([self.id_to_node[i].depth for i in filling])
                
        node = ConceptNode(node_id=node_id, node_count=node_count, type_count=type_count, node_type=node_type, depth=depth,
                           created=created, expiration=expiration, subject=subject, predicate=predicate,
                           object=object, description=description, embedding_key=embedding_pair[0], poignancy=poignancy,
                           keywords=keywords, filling=filling, embedding=embedding_pair[1])

        if node_type == 'chat':
            self.seq_chat[0:0] = [node]
            cache, cache_strength = self.kw_to_chat, None
        elif node_type == 'event':
            self.seq_event[0:0] = [node]
            cache, cache_strength = self.kw_to_event, self.kw_strength_event
        elif node_type == 'thought':
            self.seq_thought[0:0] = [node]
            cache, cache_strength = self.kw_to_thought, self.kw_strength_thought

        keywords = [i.lower() for i in keywords]
        for kw in keywords:
            if kw in cache:
                cache[kw][0:0] = [node]
            else:
                cache[kw] = [node]

            if cache_strength is not None and f"{predicate} {object}" != "is idle":
                if kw in cache_strength:
                    cache_strength[kw] += 1
                else:
                    cache_strength[kw] = 1

        self.embeddings[embedding_pair[0]] = embedding_pair[1]

        return node
            
        
    def get_summarized_latest_events(self, retention):
        ret_set = set()
        for e_node in self.seq_event[:retention]:
            ret_set.add(e_node.spo_summary())
        return ret_set

    def get_str_seq_events(self):
        ret_str = ""
        for count, event in enumerate(self.seq_event):
            ret_str += f'{"Event", len(self.seq_event) - count, ": ", event.spo_summary(), " -- ", event.description}\n' # returns a string of tuple
        return ret_str    

    def get_str_seq_thoughts(self):
        ret_str = ""
        for count, event in enumerate(self.seq_event):
            ret_str += f'{"Thought", len(self.seq_event) - count, ": ", event.spo_summary(), " -- ", event.description}\n' # returns a string of tuple
        return ret_str  

    def get_str_seq_chats(self):
        ret_str = ""
        for event in self.seq_chat:
            ret_str += f"with {event.object.content} ({event.description})\n"
            ret_str += f"{event.created.strftime('%B %d, %Y, %H:%M:%S')}\n"
            for row in event.filling:
                ret_str += f"{row[0]}: {row[1]}\n"
        return ret_str
            

    def retrieve_relevant_thoughts(self, s, p, o):
        contents = [s, p, o]
        ret = []
        for i in contents:
            if i in self.kw_to_thought:
                ret += self.kw_to_thought[i.lower()]
        return set(ret)

    def retrieve_relevant_events(self, s, p, o):
        contents = [s, p, o]
        ret = []
        for i in contents:
            if i in self.kw_to_event:
                ret += self.kw_to_event[i.lower()]
        return set(ret)
    