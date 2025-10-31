import random
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization is used (0 means no prioritization, 1 full prioritization)
        self.buffer = []  # Holds the transitions (states, actions, rewards, next_states)
        self.priorities = []  # Holds the priorities for each transition
        self.pos = 0  # Position for overwriting when buffer is full
        self.max_priority = 1.0  # Initialize max priority for new transitions

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, priority=None):
        if priority is None:
            #print("Calculating priority for the new experience!")
            priority = self.max_priority  # Use the highest priority for new transitions
        #print(f"length of buffer: {len(self.buffer)}")
        if len(self.buffer) < self.capacity:
            #print("Adding experience to the buffer!")
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            # Overwrite old transitions (circular buffer)
            #print("Overwriting experience in the buffer!")
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
        
        self.pos = (self.pos + 1) % self.capacity
# #priority is based on the TD error - probabilistic sampling method

    def sample(self, batch_size, beta):
        if len(self.buffer) == 0:
            return None

        scaled_priorities = np.array(self.priorities) ** self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()

        # Sample unique indices based on probabilities #THIS IS THE CHANGE FOR PROBABILTITY BASED SAMPLING
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=sampling_probabilities)

        transitions = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * sampling_probabilities[indices]) ** (-beta)  # Importance sampling weights
        weights = weights / weights.max()  # Normalize weights

        # Separate the sampled transitions into states, actions, rewards, and next_states
        states, actions, rewards, next_states = zip(*transitions)
        
        return states, actions, rewards, next_states, indices, weights


    def update_priorities(self, indices, td_errors):
        #print("UPDATING PRIORITIES OF EXPERIENCES IN THE BUFFER!")
        #print(f"Indices: {indices}")
        #print(f"Priorities: {td_errors}")
        for idx, td_error in zip(indices, td_errors):
            #print(f"Updating priority of index {idx} to {td_error}")
            priority = abs(td_error) + 1e-5  # Small value to avoid zero priorities
            #print(f"Priority: {priority}")
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)  # Update the max priority

        #print(f"Updated priorities: {self.priorities}")
        #print(f"Max priority: {self.max_priority}")

# import random
# import numpy as np

# class PrioritizedReplayBuffer:
#     def __init__(self, capacity, alpha):
#         self.capacity = capacity
#         self.alpha = alpha
#         self.buffer = []
#         self.priorities = []
#         self.pos = 0
#     # Return the current number of elements in the buffer
#     def __len__(self):
#         return len(self.buffer) 

#     def add(self, transition, priority):
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(transition)
#             self.priorities.append(priority)
#         else:
#             self.buffer[self.pos] = transition
#             self.priorities[self.pos] = priority
#         self.pos = (self.pos + 1) % self.capacity
# #priority is based on the TD error - probabilistic sampling method
#     def sample(self, batch_size, beta):
#         if len(self.buffer) == 0:
#             return None
#         #print("PROBABILISTIC SAMPLING OF EXPERIENCES FROM THE BUFFER!")
#         #print(f"alpha = {self.alpha}, beta = {beta}")
#         #print(f"Priorities: {self.priorities}")
#         scaled_priorities = np.array(self.priorities) ** self.alpha
#         #print(f"Scaled Priorities: {scaled_priorities}")
#         sampling_probabilities = scaled_priorities / scaled_priorities.sum()
#         #print(f"Sampling Probabilities: {sampling_probabilities}")

#         indices = random.choices(range(len(self.buffer)), k=batch_size, weights=sampling_probabilities)
#         #print(f"Indices: {indices}")
#         transitions = [self.buffer[idx] for idx in indices]
#         #print(f"Transitions: {transitions}")

#         total = len(self.buffer)
#         #print(f"Total: {total}")
#         weights = (total * sampling_probabilities[indices]) ** (-beta)
#         #print(f"Weights: {weights}")
#         weights = weights / weights.max()
#         #print(f"Normalized Weights: {weights}")
#         #print("PROBABILISTIC SAMPLING COMPLETED!")

#         states, actions, rewards, next_states = zip(*transitions)
#         return states, actions, rewards, next_states, indices, weights

#     def update_priorities(self, indices, priorities):
#         #print("UPDATING PRIORITIES OF EXPERIENCES IN THE BUFFER!")
#         #print(f"Indices: {indices}")
#         #print(f"Priorities: {self.priorities}")
#         for idx, priority in zip(indices, priorities):
#             #print(f"Updating priority of index {idx} to {priority}")
#             self.priorities[idx] = priority
