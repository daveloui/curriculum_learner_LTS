import heapq
import numpy as np

from models.memory import Trajectory
import copy

class TreeNode:
    def __init__(self, parent, game_state, p, g, levin_cost, action):
        self._game_state = game_state
        self._p = p
        self._g = g
        self._levin_cost = levin_cost
        self._action = action
        self._parent = parent
        self._probabilitiy_distribution_a = None
    
    def __eq__(self, other):
        """
        Verify if two tree nodes are identical by verifying the 
        game state in the nodes. 
        """
        return self._game_state == other._game_state
    
    def __lt__(self, other):
        """
        Function less-than used by the heap
        """
        return self._levin_cost < other._levin_cost    
    
    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self._game_state.__hash__()
    
    def set_probability_distribution_actions(self, d):
        self._probabilitiy_distribution_a = d
        
    def get_probability_distribution_actions(self):
        return self._probabilitiy_distribution_a
    
    def set_levin_cost(self, c):
        self._levin_cost = c
    
    def get_p(self):
        """
        Returns the pi cost of a node
        """
        return self._p
    
    def get_g(self):
        """
        Returns the pi cost of a node
        """
        return self._g
    
    def get_game_state(self):
        """
        Returns the game state represented by the node
        """
        return self._game_state
    
    def get_parent(self):
        """
        Returns the parent of the node
        """
        return self._parent
    
    def get_action(self):
        """
        Returns the action taken to reach node stored in the node
        """
        return self._action
        

class BFSLevinMult():
    
    def __init__(self, use_heuristic=True, use_learned_heuristic=False, k_expansions=32):
        self._use_heuristic = use_heuristic
        self._use_learned_heuristic = use_learned_heuristic
        self._smallest_float = np.nextafter(0, 1)
        self._k = k_expansions
    
    def get_levin_cost(self, child_node, predicted_h):
        
        d = child_node.get_g()
        d_over_pi = d / child_node.get_p()
        
        if self._use_learned_heuristic and self._use_heuristic:
            max_h = max(predicted_h, child_node.get_game_state().heuristic_value())
            estimated_future_expansions = max_h / ((child_node.get_p()) ** ((max_h + d) / d))
        elif self._use_learned_heuristic:
            if predicted_h < 0:
                predicted_h = 0
            estimated_future_expansions = predicted_h / ((child_node.get_p()) ** ((predicted_h + d) / d) + self._smallest_float)
        else:
            h_value = child_node.get_game_state().heuristic_value()
            estimated_future_expansions =  h_value / ((child_node.get_p()) ** ((h_value + d) / d) + self._smallest_float)
            
        return d_over_pi + estimated_future_expansions


    def search(self, data):
        """
        Performs Best-First LTS . 
        
        Returns solution cost, number of nodes expanded, and generated
        """
        state = data[0] 
        nn_model = data[1]
        
        _open = []
        _closed = set()
        
        expanded = 0
        generated = 0
        
        if self._use_learned_heuristic:
            _, action_distribution, _ = nn_model.predict(np.array([state.get_image_representation()]))
        else:
            _, action_distribution = nn_model.predict(np.array([state.get_image_representation()]))
        
        node = TreeNode(None, state, 1, 0, 0, -1)
        node.set_probability_distribution_actions(action_distribution[0])
        
        heapq.heappush(_open, node)
        _closed.add(state)
        
        predicted_h = np.zeros(self._k)
        
        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated= []
        
        while len(_open) > 0:
            
            node = heapq.heappop(_open)                            
                
            expanded += 1
            actions = node.get_game_state().successors_parent_pruning(node.get_action())
            probability_distribution = node.get_probability_distribution_actions()
                            
            for a in actions:
                child = copy.deepcopy(node.get_game_state())
                child.apply_action(a)

                if child.is_solution(): 
                    return node.get_g() + 1, expanded, generated
                
                if child in _closed:
                    continue
                
                child_node = TreeNode(node, child, node.get_p() * probability_distribution[a], node.get_g() + 1, -1, a)

                children_to_be_evaluated.append(child_node)
                x_input_of_children_to_be_evaluated.append(child.get_image_representation())
                
                if len(children_to_be_evaluated) == self._k or len(_open) == 0:

                    if self._use_learned_heuristic:
                        _, action_distribution, predicted_h = nn_model.predict(np.array(x_input_of_children_to_be_evaluated))
                    else:
                        _, action_distribution = nn_model.predict(np.array(x_input_of_children_to_be_evaluated))
                    
                    for i in range(len(children_to_be_evaluated)):
                        generated += 1
                        
                        levin_cost = self.get_levin_cost(children_to_be_evaluated[i], 
                                                        predicted_h[i])
                        children_to_be_evaluated[i].set_probability_distribution_actions(action_distribution[i])
                        children_to_be_evaluated[i].set_levin_cost(levin_cost)
        
                        
                        if children_to_be_evaluated[i].get_game_state() not in _closed:
                            heapq.heappush(_open, children_to_be_evaluated[i])
                            _closed.add(children_to_be_evaluated[i].get_game_state())
                        
                    children_to_be_evaluated.clear()
                    x_input_of_children_to_be_evaluated.clear()    
        
    def _store_trajectory_memory(self, tree_node, expanded):
        """
        Receives a tree node representing a solution to the problem. 
        Backtracks the path performed by search, collecting state-action pairs along the way. 
        The state-action pairs are stored alongside the number of nodes expanded in an object of type Trajectory,
        which is added to the variable memory. 
        """
        states = []
        actions = []
        solution_costs = []
        
        state = tree_node.get_parent()
        action = tree_node.get_action()
        cost = 1
        
        while not state.get_parent() is None:
            states.append(state.get_game_state())
            actions.append(action)
            solution_costs.append(cost)
            
            action = state.get_action()
            state = state.get_parent()
            cost += 1
            
        states.append(state.get_game_state())
        actions.append(action)
        solution_costs.append(cost)
        
        return Trajectory(states, actions, solution_costs, expanded, tree_node.get_p())        
    
    
    def search_for_learning(self, data):
        """
        Performs Best-First LTS bounded by a search budget.
        
        Returns Boolean indicating whether the solution was found,
        number of nodes expanded, and number of nodes generated
        """
        state = data[0]
        puzzle_name = data[1]
        budget = data[2]
        nn_model = data[3]
        
        _open = []
        _closed = set()
        
        expanded = 0
        generated = 0
        
        if self._use_learned_heuristic:
            _, action_distribution, _ = nn_model.predict(np.array([state.get_image_representation()]))
        else:
            _, action_distribution = nn_model.predict(np.array([state.get_image_representation()]))
        
        node = TreeNode(None, state, 1, 0, 0, -1)
        node.set_probability_distribution_actions(action_distribution[0])
        
        heapq.heappush(_open, node)
        _closed.add(state)
        
        predicted_h = np.zeros(self._k)
        
        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated= []
        
        while len(_open) > 0:
            
            node = heapq.heappop(_open)                            
            
            expanded += 1
            actions = node.get_game_state().successors_parent_pruning(node.get_action())
            probability_distribution = node.get_probability_distribution_actions()
            
            if expanded >= budget:
                return False, None, expanded, generated, puzzle_name
                            
            for a in actions:
                child = copy.deepcopy(node.get_game_state())
                child.apply_action(a)

                child_node = TreeNode(node, child, node.get_p() * probability_distribution[a], node.get_g() + 1, -1, a)

                if child.is_solution(): 
                    trajectory = self._store_trajectory_memory(child_node, expanded)
                    return True, trajectory, expanded, generated, puzzle_name
                

                children_to_be_evaluated.append(child_node)
                x_input_of_children_to_be_evaluated.append(child.get_image_representation())
                
                if len(children_to_be_evaluated) == self._k or len(_open) == 0:

                    if self._use_learned_heuristic:
                        _, action_distribution, predicted_h = nn_model.predict(np.array(x_input_of_children_to_be_evaluated))
                    else:
                        _, action_distribution = nn_model.predict(np.array(x_input_of_children_to_be_evaluated))
                    
                    for i in range(len(children_to_be_evaluated)):
                        generated += 1
                        
                        levin_cost = self.get_levin_cost(children_to_be_evaluated[i],  
                                                        predicted_h[i])
                        children_to_be_evaluated[i].set_probability_distribution_actions(action_distribution[i])
                        children_to_be_evaluated[i].set_levin_cost(levin_cost)
        
                        
                        if children_to_be_evaluated[i].get_game_state() not in _closed:
                            heapq.heappush(_open, children_to_be_evaluated[i])
                            _closed.add(children_to_be_evaluated[i].get_game_state())
                        
                    children_to_be_evaluated.clear()
                    x_input_of_children_to_be_evaluated.clear()      
