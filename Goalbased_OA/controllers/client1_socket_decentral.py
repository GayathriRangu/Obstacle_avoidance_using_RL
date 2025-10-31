"""DQN_FL_30 nov 23 INPUTS NORMALIZED AND BIAS INCLUDED IN AGGREGATION"""
from msvcrt import kbhit
import os
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from controller import Robot, Supervisor
import math
import pickle
from per_buffer import PrioritizedReplayBuffer
import copy
import threading
import pandas as pd
import io
import struct
import csv
import socket
import new_immuno as isa
import queue
import random

# Initialize a queue to store the return value from the thread
model_queue = queue.Queue()
send_ep_queue=queue.Queue()
#Since threads don’t return values directly, can use a queue.Queue as a mechanism to get the result back from the thread!!


log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)
# Create log files
q_values_log_file = os.path.join(log_dir, "q_values_log.csv")
td_error_log_file = os.path.join(log_dir, "td_error_log.csv")
loss_log_file = os.path.join(log_dir, "loss_log.csv")
priority_log_file = os.path.join(log_dir, "priority_log.csv")
# Define the directory and file for logging buffer data
buffer_log_dir = "buffer_logs"
os.makedirs(buffer_log_dir, exist_ok=True)
buffer_log_file = os.path.join(buffer_log_dir, "replay_buffer_log.csv")

TIME_STEP =100 #in millisconds #increase / decrease for slowing speeding up the simulation
MAX_SPEED = 6.28
send_weights_iter=100 #change this to change the num of iterations afgter which the weights are sent
last_loss=None
window_size=1000 # window size increase
loss_window=[]

robot = Supervisor()
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)
prev_pos = gps.getValues()
print("Starting position:", prev_pos)
#initialize devices
ps=[0,0,0,0,0]
psNames=["ps0", "ps2", "ps4","ps5","ps7"] #two front ones 7,0; two side ones 5,2; one back 4
ps=[robot.getDevice(name) for name in psNames]
sampling_period=50 #in ms  It determines how often the proximity sensors will be sampled to get their values. A smaller value means the sensors will be read more frequently, while a larger value reduces the frequency.
for sensor in ps:
    sensor.enable(sampling_period)

# Initialize motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)
psValues = [0,0,0,0,0]
global_model_better=0 #initially set to 0
Ravg=0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
gps = robot.getDevice('gps')
gps.enable(TIME_STEP)
goal_pos = [-0.0216616, 0.450585, 0.0488352] # (x, z) coordinates of the goal in meters – adjust as needed
TARGET_THRESHOLD = 0.2
X_MIN, X_MAX = -0.42, 0.42
Z_MIN, Z_MAX = -0.42, 0.42
Y_POS = 0.035  # e-puck height
reach_count=0
# Get robot position field
# translation_field = robot.getSelf().getField("translation")

# # Get robot node by DEF
# robot_node = robot.getFromDef("EPUCK")  # you need to DEF your robot in the world
# translation_field = robot_node.getField("translation")
# # Get target point node and its translation field
# target_node = robot.getFromDef("target_point")
# target_field = target_node.getField("translation")
# target_pos = target_field.getSFVec3f()
  
# Initialize Tarpy and agent behavior
lock=threading.Lock()

weights_fc1_local=[]#platform weights &BIASES
weights_fc2_local=[]
weights_fc3_local=[]
bias_fc1_local=[]
bias_fc2_local=[]
bias_fc3_local=[]

weights_fc1_global=[]#agent weights
weights_fc2_global=[]
weights_fc3_global=[]
bias_fc1_global=[]
bias_fc2_global=[]
bias_fc3_global=[]
q_con_table=[]
model_age=0 #2*num_of_nodes_in_the_network #impleis two rounds
aggregation_count=0 #2*num_of_nodes_in_the_network #implies two rounds


def distance(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def random_position_within_arena():
    """Generate a safe random spawn position inside arena bounds"""
    x = random.uniform(X_MIN, X_MAX)
    z = random.uniform(Z_MIN, Z_MAX)
    return [x, Y_POS, z]


# Initialize CSV writers
def init_csv_file(file_path, headers):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

# Helper function to log data
def log_to_csv(file_path, data):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


# Initialize CSV file for replay buffer
def init_buffer_csv(file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Add headers (customize based on what your buffer stores)
        writer.writerow(["iteration", "state", "action", "reward", "next_state", "priority"])

# Function to log the replay buffer every 500 iterations
def log_replay_buffer(iteration, replay_buffer):
    with open(buffer_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Iterate through the buffer and log the contents
        for idx in range(len((replay_buffer.buffer))):  # Assuming replay_buffer has a .buffer attribute
            transition = replay_buffer.buffer[idx]
            priority = replay_buffer.priorities[idx] if idx < len(replay_buffer.priorities) else None
            state, action, reward, next_state = transition  # Adjust this based on your buffer's structure
            writer.writerow([iteration, state, action, reward, next_state, priority])

# Function to log round and calibres (5 values) in the same row
def log_round_calibres(round_num, calibres):
    with open("calibres_log.txt", 'a') as logf:
        logf.write(f"Round {round_num}:\t" + "\t".join(map(str, calibres)) + "\n")
                   
# Function to log max_conc_lists received
def log_max_conc_lists(round_num, lists_received):
    with open("max_conc_lists.txt", 'a') as logf:
        logf.write(f"Round {round_num}:\n")
        for i, conc_list in enumerate(lists_received):
            logf.write(f"Client {i+1} list: {conc_list}\n")
        logf.write("\n")

init_csv_file(q_values_log_file, ["episode", "q_value_src", "q_value_target"])
init_csv_file(td_error_log_file, ["episode", "td_error"])
init_csv_file(loss_log_file, ["episode", "loss"])
init_csv_file(priority_log_file, ["episode", "priorities"])        
init_buffer_csv(buffer_log_file)


def aggregate_model(list1, list2):
    # Check if both inputs are lists
    if isinstance(list1, list) and isinstance(list2, list): #if its a list -- chcek size and then apply recursion
        # Check if the lists have the same length
        if len(list1) != len(list2):
            raise ValueError("Input lists must have the same length")
        
        # Recursively apply elementwise_average to each pair of elements
        return [aggregate_model(sublist1, sublist2) for sublist1, sublist2 in zip(list1, list2)]
    else:
        # Base case: if not lists, calculate the average
        return (list1 + list2) / 2 #if not list, then just return the average

def aggregate_biases(biases_list1, biases_list2): #added 30_11_23
    # Check if both inputs are lists
    if isinstance(biases_list1, list) and isinstance(biases_list2, list):
        # Check if the lists have the same length
        if len(biases_list1) != len(biases_list2):
            raise ValueError("Input lists must have the same length")
        
        # Elementwise average for bias lists
        return [(bias1 + bias2) / 2 for bias1, bias2 in zip(biases_list1, biases_list2)]
    else:
        raise ValueError("Input biases must be lists.")
   
def log_it():
    with open(file_n,'w') as logf:
                logf.write("\n------------------A global model is better than my local model--------------------\n")
    with open(file_name,'w') as lf:
        lf.write("\n------------------A global model is better than my local model--------------------\n")
with open("agent_log.txt",'w') as ll:
    #ll.write("Logging agents take decision info\n")
    ll.write("Episode ")
    ll.write("overall_diversity_agent ")
    ll.write("overall_diversity_node ")
    ll.write("imprvement=agent-node ")
    ll.write("Decision_no:1-localmodelbetter_2-globalmodelbetter ")



def take_decision(episode,round,calibre,maxconcent_list,models_list):
    local_calibre=calibre[0]
    global_calibre=calibre[1] #the one that is less is the better one!
    aggregated_model = copy.deepcopy(models_list[0]) #INITIALIZE THE AGGREGATED MODEL!
    # Zero out the parameters of the aggregated model to start averaging, sets all parameters of the aggregated_model to zero to ensure no bias during averaging
    for param in aggregated_model.parameters():
        param.data = torch.zeros_like(param.data)  # Non in-place operation to zero out the parameters  -using in-place may be efficient but since this code involves modifying the model, copying etc, it throws error
    
    with open("agent_log.txt",'a') as ll:
        ll.write(str(episode)+"  ")
        ll.write(str(calibre)+"  ")
        ll.write(str(maxconcent_list)+"  ")
        ll.write(str(models_list)+"  \n************************************************\n\n")
        if global_calibre<local_calibre: #global is better! change local with aggregation!
            print("CASE 1 !global is better! change local with aggregation!!!!!!!!")
            decision_no=1
            log_it()
            ll.write(str(episode)+" "+str(round)+" "+str(decision_no)+" globalmodelbetter")
            #update the agent models age- because its not being disturbed and is being used for aggregation
            # model_age=??????? add another attribute in client sending and receiving!
            # model_age+=1
            num_models = 2 #only two are aggregated at a time!
            #aggregating the models!!
            print(f"num models that are getting aggregated right now are {num_models}")
            for model in models_list:
                for param_agg, param_model in zip(aggregated_model.parameters(), model.parameters()):
                    param_agg.data = param_agg.data + (param_model.data / num_models) #note: donot use in-place operations to do this, it might look easy but throws errors since we are copying the agg model inbetween
            gmodel_weights_f = f"aggreg_model_weights_round_{round}.pt"
            save_model_weights(aggregated_model, gmodel_weights_f) 
            #
        else: #local is better! change the agent with the local model!
            print("CASE 2!! local is better! change agent with local model!!!!!!!!") #NOTE: IF THE GLOBAL IS AGGREGATED THEN COPYING THE TABLE  AS IS CAUSES INCONSISTENCY-- TRY OUT JUST COPYING LOCAL TO GLOBAL RATHER THAN AGGREGATION!
            decision_no=2
            ll.write(str(episode)+" "+str(round)+" "+str(decision_no)+" localmodelbetter")
            # aggregation_count= get it from the socket
            # aggregation_count+=1
            #SOL 1: COPY AGGREGATED MODEL ONTO THE AGENT -- BUT THE LIST IS THAT OF THE LOCAL AGENT AND NOT THE AGGREGATED MODEL - INCONSISTENCY
            #SOL 2: COPY THE LOCAL MODEL ONTO THE AGENT -- THIS IS THE BEST SOLUTION -- NO INCONSISTENCY
            #sol 1:
            num_models=2
            print(f"num models that are getting aggregated right now are {num_models}")
            for model in models_list:
                for param_agg, param_model in zip(aggregated_model.parameters(), model.parameters()):
                    param_agg.data = param_agg.data + (param_model.data / num_models) 
            gmodel_weights_f = f"aggreg_model_weights_round_{round}.pt"
            save_model_weights(aggregated_model, gmodel_weights_f) 
            #FOR SOL 2: copy the local model onto param_agg as is!
        return aggregated_model, decision_no

# Define the DQN neural network architecture
#tweak nn architecture #readmore on the layers used-- pertaining to the standard implementation?
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#Function to read sensor values from the robot
def read_sensors():
    psValues = []
    psValues=[sensor.getValue() for sensor in ps]
    return psValues

#FUNCTION TO CONVERT THE CONTINUOUS SENSOR VALUES INTO BINS
def bin_sensor_values(sensor_values):
    no_obstacle_thresh=75
    obstacle_proximity_thresh=175
    obstacle_close_thresh=1000
    bin_sensors=[]
    for val in sensor_values:
        #code for if val < 75 then  val=0 elif 75<=val<175 then val=1 elif 175<=val<1000 then val=2 elif val>1000 then val=3
        if val<no_obstacle_thresh:
            val=0
        elif val>=no_obstacle_thresh and val<obstacle_proximity_thresh:
            val=1
        elif val>=obstacle_proximity_thresh and val<obstacle_close_thresh:
            val=2
        elif val>=obstacle_close_thresh:
            val=3
        bin_sensors.append(val)

    return bin_sensors

def normalize_sensor_values(bin_sensor_values):
    # Find the minimum and maximum values in the bin_sensor_values
    min_val = 0
    max_val = 3

    # Normalize each value in bin_sensor_values to the range [0, 1]
    normalized_values = [(val - min_val) / (max_val - min_val) for val in bin_sensor_values]

    return normalized_values
# Define the path to the log file
log_file_path = "batch_logs.txt"

file=open("loss_window.txt","w")
file.write("average gradients")
file.write(" ")
file.write("epsilon\n")
# Function to select an action using epsilon-greedy policy
def select_action(state, epsilon): #changed during the testing processs
    if np.random.rand() < epsilon:
      #  print("random action")
        return np.random.choice(output_size), 0
    else:
        with torch.no_grad(): #disable gradient calculation #only forward prop
     #       print("NN Based Action")
            return model(state).argmax().item(), model(state).max().item()#return the index of the maximum value in the tensor #returnmax q value also added newly
        #note down the q value that this returns
def select_action_test(state, epsilon,model): #changed during the testing processs
    if np.random.rand() < epsilon:
      #  print("random action")
        return np.random.choice(output_size), 0
    else:
        with torch.no_grad(): #disable gradient calculation #only forward prop
     #       print("NN Based Action")
            return model(state).argmax().item(), model(state).max().item()#return the index of the maximum value in the tensor #returnmax q value also added newly
        #note down the q value that this returns


# to control the robot's movement
def move_forward():
    leftSpeed = 0.5 * MAX_SPEED
    rightSpeed = 0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    #print("forward")
    time.sleep(0.1)

def backward():
    leftSpeed = -0.2 * MAX_SPEED
    rightSpeed = -0.2 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    #print("backward")
    time.sleep(0.1)


def left():
    leftSpeed = -0.5 * MAX_SPEED
    rightSpeed = 0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    #print("left")
    time.sleep(0.1)

def right():
    leftSpeed = 0.5 * MAX_SPEED
    rightSpeed = -0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    #print("right")
    time.sleep(0.1)



def calculate_reward(previous_state, current_state, action, current_pos, previous_pos, goal_pos):
    global reach_count
    reward=0
    # Loop through each sensor and check for transitions
    for prev_val, curr_val in zip(previous_state, current_state):
        #0->0 +10; 0->1 +1; 0->2 -5; 0->3 -10;
        #1->0 +5; 1->1 +1; 1->2 -5; 1->3 -10;
        #2->0 +10; 2->1 +5; 2->2 -5; 2->3 -10;
        if prev_val==0 and curr_val==0 and all(curr==0 for curr in current_state)==0:
            reward+=2
            #print("1. +10")
        elif prev_val==0 and curr_val==1:
            reward+=0.1
            #print("2. +1")
        elif prev_val==0 and curr_val==2:
            reward-=1
            #print("3. -5")
        elif prev_val==0 and curr_val==3:
            reward-=2
            #print("4. -10")
            
        elif prev_val==1 and curr_val==0:
            reward+=1
            #print("5. +5")
        elif prev_val==1 and curr_val==1:
            reward+=0.1
            #print("6. +1")
        elif prev_val==1 and curr_val==2:
            reward-=1
            #print("7. -5")
        elif prev_val==1 and curr_val==3:
            reward-=2
            
            #print("8. -10")
        elif prev_val==2 and curr_val==0:
            reward+=2
            #print("9. +10")
        elif prev_val==2 and curr_val==1:
            reward+=1
           # print("10. +5")
        elif prev_val==2 and curr_val==2:
            reward-=1
          #  print("11. -5")
        elif prev_val==2 and curr_val==3:
            reward-=2
         #   print("12. -10")
            
        #3->0 +10; 3->1 +5; 3->2 +1; 3->3 -10;
        elif prev_val==3 and curr_val==0:
            reward+=2
        #    print("13. +10")
        elif prev_val==3 and curr_val==1:
            reward+=1
       #     print("14. +5")
        elif prev_val==3 and curr_val==2:
            reward+=0.1
      #      print("15. +1")
        elif prev_val==3 and curr_val==3:
            reward-=2
     #       print("16. -10")
            
        # if any(curr ==3 for curr in current_state): #new changeeeeeee
        #     reward=0
        #     reward-=10
        #     break
        if action ==0 and current_state==[0,0,0,0,0]:
            reward+=2
      # ====== NEW ADDITION: movement towards goal reward ======
    current_pos = gps.getValues()
    print("Current position:", current_pos)
    dist_to_target = distance(current_pos, goal_pos)
    prev_dist_to_target = distance(previous_pos, goal_pos)
    print("Distance to target:", dist_to_target)
    # Reward for getting closer
    delta_dist = prev_dist_to_target - dist_to_target  # positive if moving closer
    distance_reward = delta_dist * 10  # scale factor, tune as needed
    reward += distance_reward
    # Compute 2D distance to target
    if dist_to_target < TARGET_THRESHOLD:
        print("###################################################################################################Reached target:", goal_pos)
        new_pos = random_position_within_arena()
        robot.getSelf().getField("translation").setSFVec3f(new_pos)
        robot.getSelf().getField("rotation").setSFRotation([0, 1, 0, 0])  # No tilt
        # time.sleep(0.1)  # Allow position update
        robot.step(TIME_STEP)
        reward+=20
        reach_count+=1
        # Keep it green for ~1s, then revert
        for _ in range(15):  # 15 * 64ms ≈ 1 second
            robot.step(TIME_STEP)
    # Reward: closer = higher

    print(f"Reach_count: {reach_count} Distance: {dist_to_target:.2f}, Reward: {reward:.2f}")

    # =========================================================


    return reward

def update_epsilon(epsilon,epsilon_min,epsilon_dec_iter,episode):    
            #update_platform_predicates(model,episode,q_con_table, avg_conc)
        # num=math.log(0.1)/math.log(0.995) =459 this is when the epsilon =0.1 #is done for 4000 iterations
    subval=(1-epsilon_min)/epsilon_dec_iter
    if episode<=epsilon_dec_iter:
        epsilon=max(epsilon-subval, epsilon_min)
        #epsilon = max(epsilon * epsilon_decay, epsilon_min) #0.995 ?? decay value?? hyperparam decay value
    elif episode>epsilon_dec_iter:
        epsilon=0.05
    return epsilon

#SOCKET PROGRAMMING CODE --- STARTS HERE -----7SEP24
def recvall(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data
def send_models(client_socket,model,max_conc_list):
     #mention the server ip and port -- this is to connecy to the server #next node ip, [port]
    buffer = io.BytesIO()
    print("         ======================SENDING MODELS TO THE NEXT NODE!======================")
    torch.save(model, buffer)
    serialized_model = buffer.getvalue()
    # List

    #data_list = [i for i in range(1024)]  # initialize with random values
    #print(f"SENDING max_conc list..... {max_conc_list}")
    serialized_list = pickle.dumps(max_conc_list)
    # Send the length of the serialized model
    client_socket.sendall(struct.pack('>I', len(serialized_model)))
    # Send the actual serialized model
    #print(f"SEDNING model {model}")
    client_socket.sendall(serialized_model)
    # print("Sent model to server")
    # Send the length of the serialized list
    client_socket.sendall(struct.pack('>I', len(serialized_list)))
    # Send the serialized list
    client_socket.sendall(serialized_list)
    print("ALL SENT!")
    #print("SENT MODELS T=0")
    #client_socket.close()
    # print("Sent model and list to server")
    # Receive the length of the incoming data (first 4 bytes)

def receive_models_from_prev(conn,round):
    models_received=[] #to store the models that are received
    #prev_client_sockets=[] #to store the sockets of the clients
    lists_received=[]
    print("            ========================RECEIVING MODELS AND TABLE FROM THE PREVIOUS NODE!=======================")
    #prev_client_socket,addr=server_socket.accept()
    #print(f"connection from {addr} and socket {server_socket}")
    raw_data_len=recvall(conn,4)#WE FIRST DECODE THE FIRST 4 BYTES OF THE DATA - IT CONTAINS THE SIZE OF THE DATA
    #THEN WE WILL KNOW HOW MUCH MORE TO LOOK FOR WHILE RECEIVING
    #receive the length of incpoming data (first 4 bytes)
    if not raw_data_len:
        print("Error: Did not receive message length")     
        #increase this if the data is more
    #reads upto 32KB of incoming data from the client over the connection
    # Unpack the length (4 bytes as unsigned int)
    msglen = struct.unpack('>I', raw_data_len)[0]
    # STEP 1: RECEIVE THE MODEL, DESERIALIZE ====================================
    data = recvall(conn, msglen)
    if not data:
        print("Error: Did not receive model data")
    with open("server_logs.txt", 'a') as logf:
        logf.write(f"Round {round}, Model received at {time.ctime()} of size {len(data)} bytes\n")

    model = torch.load(io.BytesIO(data))     # Deserialize the model using PyTorch
    models_received.append(model) #USE THIS MODELS_received to SELECTIVE AGGREGATE
    client_ip, client_port = conn.getpeername()
    lmodel2_weights_f = f"round_{round}_mweights_{client_ip}.pt"
    save_model_weights(model, lmodel2_weights_f)
    print(f"Recieved model of shape {model}")
    # STEP 2 : RECEIVE THE MAX CONC LIST, DESERIALIZE ===========================
    raw_list_len = recvall(conn, 4) #FIRST GET THE LENGTH OF THE LIST
    if not raw_list_len:
        print("Error: Did not receive list length")
    list_len = struct.unpack('>I', raw_list_len)[0]
    max_conc_list = recvall(conn, list_len) #THEN GET THE LIST DATA
    if not max_conc_list:
        print("Error: Did not receive list data")
    data_list = pickle.loads(max_conc_list)
    lists_received.append(data_list) #USE THIS MAX_CONC_LIST LISTS TO FIND OUT THE CALIBRES!
    #print(f"RECEIVED LIST FROM {client_ip}")
    #print(f"LIST RECEIVED: {data_list}")
    #print(f"Received list from {addr}")
    # server_socket.close()
    # prev_client_socket.close()
    return models_received, lists_received

def client(conn, round, model, max_conc_list, episode):
    models_list = []  # [local, neighbor]
    maxconcent_list = []
    aggregation_count = 0
    model_age = 0
    decision_no=0

    # Initialize the aggregated model
    aggregated_model = copy.deepcopy(model)

    # Zero out the parameters of the aggregated model to start averaging
    for param in aggregated_model.parameters():
        param.data = torch.zeros_like(param.data)

    # Append the local model and concentration list to the lists
    models_list.append(model)  # appended local model
    maxconcent_list.append(max_conc_list)  # appended local list

    # Receive model and concentration list from the previous node
    models_received, lists_received = receive_models_from_prev(conn, round)
    
    if models_received and lists_received:
        models_list.append(models_received[0])  # appended neighbor model
        maxconcent_list.append(lists_received[0])  # appended neighbor list

        # print("Max concentration list:", maxconcent_list)
        # print("Models list:", models_list)

        # Calculate calibre
        calibre = isa.calibre_calculation(maxconcent_list, episode=round)
        print("Calibres calculated!\n")
        print(calibre)  # order of calibres in this list: [local, neighbor]

        # Log the round and all calibres
        log_round_calibres(round, calibre)
        
        # Log the max_conc_lists after receiving
        log_max_conc_lists(round, maxconcent_list)

        # Take decision based on the calibres and aggregate the models
        aggregated_model, decision_no = take_decision(episode, round, calibre, maxconcent_list, models_list)
        round=round+1
        print("decision no. is", decision_no)
    else:
        print("Error: Did not receive models or lists")

    return aggregated_model, decision_no

    
#SOCKET PROGRAMMIG ENDS HERE---
# Node-specific configurations
node_ips = ['172.16.117.133', '172.16.117.132', '172.16.117.131', '172.16.117.129', '172.16.117.128']
#node_ips=['localhost','localhost','localhost','localhost','localhost']
node_ports = [5000, 5001, 5002, 5003, 5004]  # Ports for each node
iterations_to_send=[]
node_id=0
count=0
for i in range(60): #for node 1 its 500+500*i, for node 2 its 600+500*i, for node 3 its 700+500*i, for node 4 its 800+500*i, for node 5 its 900+500*i
    num=500+(node_id*100)+i*500
    if num<30000:
        iterations_to_send.append(num)
    #print(f"ietrations_to_send: {iterations_to_send}")
print(f"ietrations_to_send: {iterations_to_send}")


# Function to save model weights to a file
def save_model_weights(model, filename):
    torch.save(model.state_dict(), filename)
#######CONCENTRATIN - BASED CODE IDIOTYPIC NETWORK 7/10/23---------------------
def initialize_q_con_table():
    print("DEF initialize q con table")
    num_states=1024 #depends on the number of sensor values used 4^5=1024
    num_actions=4
    q_con_table=[] #this is the main table. I visualized it as a 2D list. Each cell in the table is a dictionary with 2 keys: q_window and resource. 2d list of dictionaries
    for _ in range(num_states):
        state_row=[]
        for _ in range(num_actions):
            cell={'q_value': 0, 'concentration': 1, 'cell_count':0}
            #sum of the cell_count in the row will return the state_count
            state_row.append(cell)
        q_con_table.append(state_row)

    return q_con_table


#FUNCTION TO SAVE THE TABLE
logdir="qres_table_log"
if not os.path.exists(logdir):
    os.makedirs(logdir)
def save_qcon_table(q_con_table,episode):
    #print("DEF save q con table to a file")
    #CODE TO SAVE THE RESOURCE TABLE IN EACH ITERATION TO A FILE seperate
    x="q_con_table"
    table_file=os.path.join(logdir,f"{x}{episode}.pkl")
    with open(table_file,'wb') as fill:
        pickle.dump(q_con_table,fill)

#update the new q values in the table #all the q values need to be updated not just the maximum-- noteee #tested this function done unit test
#UPDATE QVALUES AND CELLCOUNT
def update_q_con_table(q_con_table,state_batch,q_value):
    actions=-1
    new_batch=convert_state_to_index(state_batch)
    i=0
    for states in new_batch:
        for actions in range(len(q_con_table[states])): #range value is 4
            q_con_table[states][actions]['q_value']=q_value[i][actions] #verify 10/10/23#verify??? DONE 5/12/23
            #if this is the maximum q value, then increase the cell count
            if q_con_table[states][actions]['q_value']==max(q_value[i]):
                #print(q_con_table[states][actions]['q_value'])#####2/12/23 CHECK THE VALUES?? IS IT INCREMENTING PROPERLY
                q_con_table[states][actions]['cell_count']+=1 #need to modify -- increase count only for the maximum one??? but it is maximum doesnt mean this is the action that was taken but this was predicted!
                #print("cell count increased",q_con_table[states][actions]['q_value'])
        i+=1
    return q_con_table #return the updated table


with open("Cavg.txt",'w') as file:
    file.write("Epsiode")
    file.write("\t")
    file.write("Overall_diversity_prev")
    file.write("\t")
    file.write("Overall_diversity_curr")
    file.write("\t")
    file.write("improvement")
    file.write("\n")
new_state_batch=[]

def get_prev_max_conc_list(q_con_table):
    max_conc_list_prev=[]
    for i in range(len(q_con_table)):
        max_q_val = max(q_entry['q_value'] for q_entry in q_con_table[i]) #for each row, find put the maximum element and its index
        max_q_index = [q_entry['q_value'] for q_entry in q_con_table[i]].index(max_q_val)
        max_conc_list_prev.append(q_con_table[i][max_q_index]['concentration'])
    return max_conc_list_prev

def intra_state_stim_supp(q_con_table, state_batch, action_batch, episode): #FUNCTION FOR INTRA STATE STIM SUPP
    improvement = 0
    max_indices_list=[]
    count=0
    state_count=[]
    max_conc_list=[]
    total_cell_count=0 
    counttemp=0
    new_batch=convert_state_to_index(state_batch) #change2
    #print(f"within the update q con function: {new_batch}")
    new_batch.sort() #this will sort the states in ascending order -just to confirm with the q_con_table order parsing
    #print(f"state batch sort is {new_batch}")
    for i in range(len(q_con_table)):
        max_q_val = max(q_entry['q_value'] for q_entry in q_con_table[i]) #for each row, find put the maximum element and its index
        max_q_index = [q_entry['q_value'] for q_entry in q_con_table[i]].index(max_q_val)
        #print(f"index of max is {max_q_index}")
        max_indices_list.append(max_q_index) #used for tracking the index places of max q values
        total_cell_count=0 #change3 make it 0 mandatorily since it is seperately calculated for each row
        for cell in q_con_table[i]: #since this will be different for each cell, we need to calculate it here inside
            total_cell_count+=copy.copy(cell['cell_count'])
            current_state_count=copy.copy(total_cell_count)
        state_count.append(total_cell_count) #used later for weighted averaging using state count in variance calculation
        if i in new_batch:#if a state is used in the batch, then only calculate the concentration for that state
            # Find the maximum Q-value in the row
            stimulation=0
            suppression=0
            temp=copy.copy(episode)
            total_state_count=temp*32 #total state count = episode number
            for j in range(len(q_con_table[i])):
                #print(f"========================j in for is {j}=====================================\n")
                q_j = copy.deepcopy(q_con_table[i][j]['q_value'])
                counttemp=copy.deepcopy(q_con_table[i][j]['cell_count'])
                m_j_i = abs(max_q_val - q_j) * (counttemp / (total_cell_count*10))#only inter-cell stim supo
                stimulation += m_j_i * q_con_table[i][j]['concentration']*0.01  #multiply cj
                suppression = 0
                suppression = m_j_i * q_con_table[i][j]['concentration']* q_con_table[i][max_q_index]['concentration'] *0.01 
                q_con_table[i][j]['concentration'] -=suppression #log this value??
                #print(f"after suppression {q_con_table[i][j]['concentration'] }")
                if q_con_table[i][j]['concentration'] < 1:# Ensure that the concentration value doesn't go below a minimum value--set lower limit
                    q_con_table[i][j]['concentration'] = 1
                if q_con_table[i][j]['concentration']>500:
                    q_con_table[i][j]['concentration']=500
            stimulation*=q_con_table[i][max_q_index]['concentration'] #alpha constanyt =0.01 #multiply cmax
            #stimulation*=q_con_table[i][max_q_index]['concentration']*0.1 #alpha constanyt =0.01 #multiply cmax
            #print(f"before stimulation {q_con_table[i][max_q_index]['concentration']}")
            q_con_table[i][max_q_index]['concentration'] += stimulation #stimulating the max cell-update its concentration #log this value
            #print(f"after stimulation {q_con_table[i][max_q_index]['concentration']}")
            ##improvement += q_con_table[i][max_q_index]['concentration']
            if q_con_table[i][max_q_index]['concentration']>500:
                q_con_table[i][max_q_index]['concentration']=500
            max_conc_list.append(q_con_table[i][max_q_index]['concentration']) #used for weighted averaging using state count in variance calculation #change1
        else: #if the state is not used in the batch, dont claculate the concentration for that state, just take its concentration value as is
            ##improvement += q_con_table[i][max_q_index]['concentration']
            max_conc_list.append(q_con_table[i][max_q_index]['concentration'])

    return q_con_table, state_count,max_conc_list,max_indices_list #return the updated table


#------------------------CONCENTRATION CODE IDIOTYPIC NETWORK---------------ENDS HERE----------------
   
#function to perform the conversion of state batch to their indices
def convert_state_to_index(state_batch):
    #print("DEF Convert state to index")
    new_state_batch=[-1]*len(state_batch) #INITIALIZE THE BATCH WITH -1 
    state_batch=state_batch.tolist()
    for i in range(len(state_batch)):
        state=state_batch[i]
        #print("state here",state)
        state=state_batch[i]
        for j in range(len(state)):
            if abs(state[j] - 0.0/3) < 1e-6:  # Check for approximate equality
                state[j] = 0
            elif abs(state[j] - 1.0/3) < 1e-6:
                state[j] = 1
            elif abs(state[j] - 2.0/3) < 1e-6:
                state[j] = 2
            elif abs(state[j] - 3.0/3) < 1e-6:
                state[j] = 3
        #print(state)
        state_index=0
        for j in range(len(state)):
            state_index+=state[j]*(4**(len(state)-j-1))
        new_state_batch[i]=state_index
        #print("state",state)
        #print("index\t"+str(new_state_batch[i]))
    
    #convert into integer
    for i in range(len(new_state_batch)):
        new_state_batch[i]=int(new_state_batch[i])

    #print(new_state_batch)
    #print("new state batch",new_state_batch)
    return new_state_batch


#RESOURCE BASED CODE ENDS-----------------------------------------------------


file_n="ALL_q_values.txt"
with open(file_n,'w') as logf:
    logf.write("episode")
    logf.write("\t")
    logf.write("all_q_src[i]")
    logf.write("\t")
    logf.write("all_q_values_target[i]")
    logf.write("\t")
    logf.write("q_src_action_value")
    logf.write("\t")
    logf.write("q_values_target_max")
    logf.write("\t")
    logf.write("q_sum_bellman")
    logf.write("\n")
file_name="s_a_r_s'_q_a'_qmax'.txt"
with open(file_name,'w') as file:
    file.write("Episode")
    file.write("\t")
    file.write("State")
    file.write("\t")
    file.write("Converted_State")
    file.write("\t")
    file.write("Action")
    file.write("\t")
    file.write("Next_State")
    file.write("\t")
    file.write("Converted_next_State")
    file.write("\t")
    file.write("Reward")
    file.write("\t")
    file.write("Qval_prev_action")
    file.write("\t")
    file.write("New_Action")
    file.write("\t")
    file.write("Qval_new_max")
    file.write("\t")
    file.write("Cavg")
    file.write("\n")
    #aggregated_model = return_queue.get()  # Blocks until a value is available
    #aggregated_model=listen_for_data()
    #listener_thread.join()  # Wait for the thread to finish
## WHEN TO IDIOTYPIC NETWORK TO BE STIMULATED N SUPPRESSED? WHILE THE TRAINING OR TESTING PHASE? IS THE TESTING PHASE STILL THERE??
#8/10/23 I THINK THE TESTING PHASE SHOULD BE THERE SINCE THAT IS WHEN HE MODEL IS PREDICTING
def train_dqn(node_id,model, target_model,tmp_model):
    # reach_count=0
    alpha=0.6 #priority exponent
    beta=0.4 #importance sampling weight exponent, increases to 1 over time
    beta_increment_per_episode=0.001
    batch_size = 32
    gamma = 0.99 ####changeeeeeee
    epsilon = 1.0
    epsilon_decay = 0.995 #linear annealing??
    epsilon_min = 0.05
    update_timer=0
    t=0
    episode=0
    epsilon_dec_iter=4000
    target_update = 10  # Update the target network every X episodes
    replay_buffer_size = 1000
    increasing_epsilon=False
    optimizer = optim.Adam(model.parameters()) #optimizer is the Adam optimizer 
    huber_loss = nn.HuberLoss(reduction='none',delta=1.0) #criterion is the Mean Squared Error (MSE) loss function #huber
    Ravg=[]
    qtarlist=[]
    qvalmean=[]
    replay_buffer = PrioritizedReplayBuffer(replay_buffer_size, alpha)
    actual_reward_list=[]
    rewards_list = []
    round=0
    qsum_list = []
    q_values_list = []
    state=[]
    next_state=[]
    episodelist=[]
    max_conc_list=[]
    losslist=[]
    epsilon_list=[]
    q_action_when_exploited=[]
    buffer_list=[]
    reward_window=[]#initialize it with all 0s
    reward_window=[0]*window_size
    random_batch_list=[]
    moving_sum_rew_list=[]
    q_upd_mean=[]
    ravg_current=0
    q_con_table=initialize_q_con_table()
    avg_conc=0
    normalized_sensors1=[]
    normalized_sensors2=[]
    total_reward = 0
    #SOCKET CODE 3/10/24
    count=0
    current_ip = node_ips[node_id]
    next_ip = node_ips[(node_id + 1) % 5]
    current_port = node_ports[node_id]
    next_port = node_ports[(node_id + 1) % 5] 
    model_lock = threading.Lock()
    send_ep=0

    def listen_for_data():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((current_ip, current_port))
            server_socket.listen(5)
            print(f"[Node {node_id + 1}] Listening for incoming data on {current_ip}:{current_port}...")
            
            while True:  # Keep the server open to listen for multiple rounds  
                conn, addr = server_socket.accept()  # Wait for a connection
                print(f"[Node {node_id + 1}] Connection established with {addr}")
                with conn:
                    #round=round+1
                    #for all aggreg ------- below
                    # aggregated_model = client(conn, round, model, max_conc_list, episode)
                    # print(f"episode value in thread is {episode}")
                    # model_queue.put(aggregated_model)
                    #==========for all aggregg -------- above
                    aggregated_model,decision_no = client(conn, round, model, max_conc_list, episode)
                    print("DECISION RETURNED IS ",decision_no)
                    print(f"episode value in thread is {episode}")
                    if decision_no==1: #if global is better, then change the lcoal model with the aggregated model, so put the agg model in the queue so that it can be updated, and this agg is taken to the next node
                        model_queue.put(aggregated_model)
                        send_ep=episode+200 #gestation  period
                        send_ep_queue.put(send_ep)
                        print(f"send_ep val updated to {send_ep}")
                    elif decision_no==2: #local model is better, since we are opting for the averaging option, and not locally copying the model, in the original paper, we will not be disturbign the local model, the agg model ois taken to the next node
                    #Nothing to put in the queue
                        print("not putting in the queue")
                        send_ep=episode+10 #gestation  period
                        send_ep_queue.put(send_ep)
                        print(f"send_ep val updated to {send_ep}")
                    #send_ep=episode+500
                    print(f"send_ep val updated to {send_ep}")
                    #after receiving the model, after 500 iterations, send the model to next


#def start_listener_thread(episode,model,max_conc_list):
    listener_thread = threading.Thread(target=listen_for_data, daemon=True)
    listener_thread.start()

    # Simulate training and wait for iteration limit
    # After iterations are complete, send the model and list to the next node
    def send_data(model,max_conc_list):
        print(f"[Node {node_id + 1}] Establishing connection to Node {node_id + 2} on {next_ip}:{next_port}...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((next_ip, next_port))
            print(f"client socket is {client_socket}")
            #print(f"[Node {node_id + 1}] Connection established with Node {node_id + 2}. Sending model and list...")
            send_models(client_socket, model, max_conc_list)
            #print(f"[Node {node_id + 1}] Sent model and list to Node {node_id + 2}.")
    # client_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # client_socket.connect(('172.16.117.132',50002))
    for episode in range(num_episodes):
        robot.step(TIME_STEP)
        
        print("===========================================================================================")
        episodelist.append(episode)
        #time.sleep(1)
        print("Episode:", episode)
        #CHECK FOR AGGREGATED_MODEL IF RECEIVED, UPDATE THE MODELS
        if not model_queue.empty():
            aggregated_model = model_queue.get()  # Retrieve and also remove the aggregated model from the queue
            model.load_state_dict(aggregated_model.state_dict())  # Update local model
            target_model.load_state_dict(aggregated_model.state_dict())
            model_weights_filename = f"AFTER_AGGREG_mw_ep_{episode}.pt"
            # send_ep=episode+200 #gestation  period
            # print(f"send_ep val updated to {send_ep}")
            save_model_weights(model, model_weights_filename)
            print(f"[Node {node_id + 1}] Aggregated model received and loaded at episode {episode}")
        #SOCKET STUFF
        state=read_sensors()
        #print("Current actual State:", state)
        bin_sensors1 = bin_sensor_values(state)
        normalized_sensors1=normalize_sensor_values(bin_sensors1)
      #  print("Current binned State:", bin_sensors1)
        state_tensor = torch.tensor(normalized_sensors1, dtype=torch.float32)
        previous_pos = gps.getValues()
        
        action, q_action = select_action(state_tensor, epsilon)
        if action == 0:
            move_forward()
        elif action == 1:
            backward()
            #stop()
        elif action == 2:
            left()
        elif action == 3:
            right()

        # Execute the action and get the next state and reward 
        robot.step(TIME_STEP) # this is for updating the robot sensor values otherwise, the current and next states come up same
        next_state = read_sensors()
        bin_sensors2=bin_sensor_values(next_state)
        normalized_sensors2=normalize_sensor_values(bin_sensors2)
        current_pos = gps.getValues()
        reward = calculate_reward(bin_sensors1,bin_sensors2, action,current_pos, previous_pos, goal_pos) #calculate the reward based on current and next state
        
        if len(reward_window)==window_size:
            reward_window.pop(0)
        reward_window.append(reward)
        


        total_reward += reward #instead take rewards over a window of time and avg them , sort of moving avg.
        #add the window buiffer
        reward_moving_avg=sum(reward_window) #this will do the sum of all the elements in the list
        q_action_when_exploited.append(q_action)
        actual_reward_list.append(reward)
        next_state_tensor = torch.tensor(normalized_sensors2, dtype=torch.float32)
        rewards_list.append(total_reward)
        moving_sum_rew_list.append(reward_moving_avg)
        # Store the transition in the replay buffer
        #replay_buffer.append((state_tensor, action, reward, next_state_tensor))
        transition = (state_tensor, action, reward, next_state_tensor)
        max_priority = max(replay_buffer.priorities) if replay_buffer.buffer else 1.0
        q_con_table_copy=copy.deepcopy(q_con_table)
        replay_buffer.add(transition,max_priority)
        #replay_buffer.add(transition, max_priority) #initially it is added with the max priority
        #replay_buffer.append((state_tensor, action, reward, next_state_tensor))

        #STOREEE WHAT DID THE REPLAY BUFFER HAD IN EVERY ITERATION #5/10/23
        normalized_sensors1=normalized_sensors2
        if episode == 200: #changeeeee
            for i in range(199):
                qtarlist_clone=q_sum.clone().detach()
                q_values_list_clone=q_values.clone().detach()
                qtarlist.extend(qtarlist_clone[i].item() for i in range(32))
                q_values_list.extend(q_values_list_clone[i].item() for i in range(32))
                losslist.append(loss.item()) 
                qsum_list.append(q_sum.mean().item())
                qvalmean.append(q_values.mean().item())
                q_upd_mean.append(q_values.mean().item())
                Ravg.append(50)
        # Train the DQN from the replay buffer
        if len(replay_buffer) >= 200: #NEEDS TO BE CHANGED TWEAKKKKKK
            sample= replay_buffer.sample(batch_size, beta)
            #batch = np.random.choice(len(replay_buffer), batch_size, replace=False) #randomly select batch_size number of elements from the replay buffer
            if sample is not None:
                state_batch, action_batch, reward_batch, next_state_batch, indices, weights = sample
                state_batch = torch.stack(state_batch)  # Use stack to create a batch tensor
                action_batch = torch.tensor(action_batch)
                next_state_batch = torch.stack(next_state_batch)
                #reward_batch = torch.tensor(reward_batch)
                # state_batch = torch.cat(state_batch)
                # action_batch = torch.tensor(action_batch)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
                #next_state_batch = torch.cat(next_state_batch)
                weights = torch.tensor(weights, dtype=torch.float32)
                #compute q-values and target q values
                q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                #log these qvalues in a text file
                all_q_src=model(state_batch).tolist()
                q_values_target = target_model(next_state_batch).max(1)[0]
                all_q_values_target=target_model(next_state_batch).tolist()
                q_sum = reward_batch + gamma * q_values_target
                
                # Clip Q-values and q_sum to prevent extreme values
                q_values = torch.clamp(q_values, -1e5, 1e5)
                q_sum = torch.clamp(q_sum, -1e5, 1e5)
                #TD ERROR
                td_errors = q_sum - q_values
                unweighted_loss = huber_loss(q_values, q_sum) #calculate the loss between q_values and q_sum temporal difference
                loss=(weights*unweighted_loss).mean() #weighted loss- applied importance sampling weights
                #????? change ????
                losslist.append(loss.item()) #####loss value along with gradients , computation graph is stored hence, .item to get the scalar value
                if len(loss_window)>=window_size:
                    loss_window.pop(0)
                loss_window.append(loss.item())
                torch.autograd.set_detect_anomaly(True)
                optimizer.zero_grad() #zero the gradients because PyTorch accumulates the gradients on subsequent backward passes
                loss.backward() #backpropagate the loss
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step() #update the weights and biases
                # Update priorities based on the TD-error
                new_priorities = td_errors.detach().abs().cpu().numpy()
                replay_buffer.update_priorities(indices, new_priorities)

                # Log Q-values
                #log_to_csv(q_values_log_file, [episode, all_q_src, all_q_values_target])
                # Log TD errors
                log_to_csv(td_error_log_file, [episode, td_errors.tolist()])
                # Log loss
                log_to_csv(loss_log_file, [episode, loss.item()])
                # Log new priorities
                log_to_csv(priority_log_file, [episode, new_priorities])

                file_n="ALL_q_values.txt"
                with open(file_n,'a') as logf:
                    for i in range(len(state_batch)):
                        logf.write(str(episode))
                        logf.write("\t")
                        for j in range(len(all_q_src[i])):
                            logf.write(str(all_q_src[i][j]))
                            logf.write(",")
                        logf.write("\t")
                        for j in range(len(all_q_values_target[i])):    
                            logf.write(str(all_q_values_target[i][j]))
                            logf.write(",")
                        logf.write("\t")
                        logf.write(str(q_values[i]))
                        logf.write("\t")
                        logf.write(str(q_values_target[i]))
                        logf.write("\t")
                        logf.write(str(q_sum[i]))
                        logf.write("\n")
                qtarlist_clone=q_sum.clone().detach()
                q_values_list_clone=q_values.clone().detach()
                qtarlist.extend(qtarlist_clone[i].item() for i in range(32))
                q_values_list.extend(q_values_list_clone[i].item() for i in range(32))
                #print("QSUM MEAN: ",q_sum.mean())
                qsum_list.append(q_sum.mean().item())
                qvalmean.append(q_values.mean().item()) 
                # Logging information for each batch into the log file
                with open(log_file_path, "w") as log_file:
                    log_file.write(f"Episode: {episode}======================\n")
                    for i in range(batch_size):
                        log_entry = f"Current State: {state_batch[i]}, " \
                                    f"Action: {action_batch[i]}, " \
                                    f"Next State: {next_state_batch[i]}, " \
                                    f"Reward: {reward_batch[i]}, " \
                                    f"Q Source Value: {q_values[i]}, " \
                                    f"Q Target Value: {q_values_target[i]}, " \
                                    f"Q Target_Bellman: {q_sum[i]}\n"
                        log_file.write(log_entry)
            
                q_values_new = model(state_batch).tolist() #since we want to store the maximum q returned by the network, as that is what being considered while exploitation.
                new_action=model(state_batch).max(1)[1] #returns the indices where the maximum has occurred
                conv_state_batch=convert_state_to_index(state_batch)
                conv_next_state_batch=convert_state_to_index(next_state_batch)
                #use this when you are updating the resource values in the resource table 
                file_name="s_a_r_s'_q_a'_qmax'.txt"
                with open(file_name,'a') as file:
                    for i in range(len(state_batch)):
                        file.write(str(episode))
                        file.write("\t")
                        file.write(str(state_batch[i].tolist()))
                        file.write("\t")
                        file.write(str(conv_state_batch[i]))
                        file.write("\t")
                        file.write(str(action_batch[i].item()))
                        file.write("\t")
                        file.write(str(next_state_batch[i].tolist()))
                        file.write("\t")
                        file.write(str(conv_next_state_batch[i]))
                        file.write("\t")
                        file.write(str(reward_batch[i].item()))
                        file.write("\t")
                        file.write(str(q_values[i].tolist()))
                        file.write("\t")
                        file.write(str(new_action[i].item()))
                        file.write("\t")
                        file.write(str(q_values_new[i]))
                        file.write("\t")
                        file.write(str(td_errors[i].item()))
                        file.write("\t")
                        file.write(str(new_priorities[i]))
                        file.write("\n")
                        file.write(str(loss.item()))
                        file.write("\n")
                q_values_prev_max = model(state_batch).max(1)[0]
                q_upd_mean.append(q_values_prev_max.mean().item())
                q_con_table=update_q_con_table(q_con_table, state_batch, q_values_new) #update the resource table with the new q values
                q_con_table, state_count,max_conc_list,max_indices_list=intra_state_stim_supp(q_con_table, state_batch, action_batch, episode) #intra-state stim supp done here  
                q_con_table_copy=copy.deepcopy(q_con_table)
                if episode%100==0:
                    save_qcon_table(q_con_table_copy,episode)
                buffer_list.append(replay_buffer) 

        #print("epsilon value",epsilon)
        epsilon=update_epsilon(epsilon,epsilon_min,epsilon_dec_iter,episode)
        #print("EPSILON IS DECAYED")
        epsilon_list.append(epsilon)
        # Update the target network
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
            # if episode>=500:
            #     print("avg_conc before updating agent variables",avg_conc)
            #     update_platform_predicates(model,episode,q_con_table, avg_conc)
            #     chk=tar.get_val("","ravg")
            #     print("checking if the value assigned properly to the agent variable",chk)
        beta = min(1.0, beta + beta_increment_per_episode)
        if episode % 500 == 0:
            log_replay_buffer(episode, replay_buffer)
        print(f"\n\nnext send ep set to: {send_ep} and current episode is {episode}\n\n")
        #if count<len(iterations_to_send)-1 and episode<28000: #not allowing aggregation process in last two thousand iterations since in other nodes, the iterations couldve completed quicker and if this tries to send , there will be an error. so to avoid that , we are adding this condition
        if episode<28000 and episode>=500:
            #if episode == iterations_to_send[count] :
            if not send_ep_queue.empty():
                send_ep=send_ep_queue.get()
            #print(f"send ep is {send_ep}")
            #############uncommnet
            # if episode == send_ep or episode==500:
            #     print("gestation over, sending to next one!")
            #     #print(f"send data , iteration {episode} and iteration to send {iterations_to_send[count]}")
            #     send_data(model,max_conc_list)
            #     count+=1
            
        # # Save model weights periodically
        if episode % 1000 == 0:
            model_weights_filename = f"mw_ep_{episode}.pt"
            save_model_weights(model, model_weights_filename)

        # Ensure the robot continues learning and moving even during socket communication delays
        #robot.step(TIME_STEP)
     # Ensure the listener thread has finished
    #listener_thread.join()
    print(f"[Node {node_id + 1}] Listener thread has finished. Round completed.")

    return episodelist, actual_reward_list, rewards_list, qsum_list, q_values_list,losslist, epsilon_list, buffer_list, random_batch_list, q_action_when_exploited, qtarlist, qvalmean, moving_sum_rew_list,Ravg,q_upd_mean



def testing_model(test_model, test_episodes,modelnum,runnum):
    t1=time.time()
    print("testing model")
    collision_count=0
    success_count=0
    window_size_t=50
    reward_window_t = []
    reward_moving_avg_t=[]
    epsilon=0.05
    with open(f"testing_ACTUAL_REW_{modelnum}_test_{runnum}",'w') as tfile:
        tfile.write("Episode\t")
        tfile.write("Actual_Reward\t")
        tfile.write("Time\n")
    with open(f"testing_CUM_REW_{modelnum}_test_{runnum}",'w') as tfile:
        tfile.write("Episode\t")
        tfile.write("Cumulative_Reward\t")
        tfile.write("Success_count\t")
        tfile.write("Collision count\t")
        tfile.write("Time\n")
    with open(f"testing_Qvalues_{modelnum}_test_{runnum}",'w') as tfile:
        tfile.write("Episode\t")
        tfile.write("Q1\t\t")
        tfile.write("Q2\t\t")
        tfile.write("Q3\t\t")
        tfile.write("Q4\t\t")
        tfile.write("Qmax\t\t")
        tfile.write("Action\t\t")
        tfile.write("Reward\t\t")
        tfile.write("Cumulative_Reward\t")
        tfile.write("Time\n")
        total_reward = 0
        actual_rewards=[]
    for episode in range(test_episodes):
        robot.step(TIME_STEP)
        state = read_sensors()
        # print("state ",state)
        bin_sensors1 = bin_sensor_values(state)
        normalized_sensors1=normalize_sensor_values(bin_sensors1)
        # print("bin_sensor1 ",bin_sensors1)
        # print("satte check ",state)
        state_tensor = torch.tensor([normalized_sensors1], dtype=torch.float32)
        previous_pos = gps.getValues()
        action, q_action = select_action_test(state_tensor, epsilon,test_model)
        if action == 0:
            move_forward()
        elif action == 1:
            backward()
            #stop()
        elif action == 2:
            left()
        elif action == 3:
            right()

        print("action ",action)
        print("qaction ",q_action)
        t2=time.time()
        t3=t2-t1
        current_pos = gps.getValues()
        reward_moving_avg_t=0
        with torch.no_grad():#since we are not training the model, we dont need to calculate the gradients hence using torch.no_grad()
            q_values_t = test_model(state_tensor)
        robot.step(TIME_STEP) # this is for updating the robot sensor values otherwise, the current and next states come up same
        next_state = read_sensors()
        for i, j in zip(state, next_state):
            if i >= 1000 and j < 1000:
                # print(f"State: {i}, Next State: {j}")
                success_count+=1
                # print("success count",success_count)
        #if any of the value in mextstate is greater than thousand and current state is also thousand then only collision
        for i, j in zip(state, next_state):
            if i >= 1000 and j >= 1000:
                # print(f"State: {i}, Next State: {j}")
                collision_count+=1
                # print("collision count",collision_count)
        print("next state",next_state)
        bin_sensors2=bin_sensor_values(next_state)
        normalized_sensor2=normalize_sensor_values(bin_sensors2)
        
        reward = calculate_reward(bin_sensors1,bin_sensors2, action, current_pos, previous_pos, goal_pos) #DOESNT MATTER IF U PASS NORMALIZED OR THESE FOR REWRADS
        print("reward ",reward)
        actual_rewards.append(reward)
        total_reward += reward 
        state = copy.copy(next_state)
        if len(reward_window_t)==window_size_t:
            reward_window_t.pop(0)
        reward_window_t.append(reward)
        reward_moving_avg_t=sum(reward_window_t)
        with open(f"testing_ACTUAL_REW_{modelnum}_test_{runnum}",'a') as tfile:
            tfile.write(str(episode))
            tfile.write("\t")
            tfile.write(str(reward))
            tfile.write("\t")
            tfile.write(str(t3))
            tfile.write("\n")
        with open(f"testing_CUM_REW_{modelnum}_test_{runnum}",'a') as tfile:
            tfile.write(str(episode))
            tfile.write("\t")
            tfile.write(str(reward_moving_avg_t))
            tfile.write("\t")
            tfile.write(str(success_count))
            tfile.write("\t")
            tfile.write(str(collision_count))
            tfile.write("\t")
            tfile.write(str(t3))
            tfile.write("\n")
        with open(f"testing_Qvalues_{modelnum}_test_{runnum}",'a') as tfile:
            tfile.write(str(episode))
            tfile.write("\t")
            for i in range(len(q_values_t)):
                #tfile.write(str(q_values_t[i][0].item()))  # Write a specific elemen
                tfile.write(str(q_values_t[i].tolist())) 
                tfile.write("\t")

            tfile.write(str(torch.max(q_values_t).item()))
            tfile.write("\t")
            tfile.write(str(action))
            tfile.write("\t")
            tfile.write(str(reward))
            tfile.write("\t")
            tfile.write(str(reward_moving_avg_t))
            tfile.write("\t")
            tfile.write(str(t3))
            tfile.write("\n")
            print(f"============{episode}=================")
            print(f"State {state}\t Action: {action} \t Next State: {next_state} \t Reward: {reward}")
            print(f"Q-values: {q_values_t}")
            print(f"Qmax: {torch.max(q_values_t)}")





#this where the execution starts                                         
input_size = 5
output_size = 4  # Assuming four possible actions (forward, backward, left, right)
num_episodes = 30000
model = DQN(input_size, output_size) #model is the DQN instance
tmp_model = DQN(input_size, output_size) #model is the DQN instance
target_model = DQN(input_size, output_size) #target_model is the target DQN instance
target_model.load_state_dict(model.state_dict()) #copy the weights and biases from model to target_model #NEWLY CHANGED 
target_model.eval() #set target_model to evaluation mode #this is to freeze the target_model while training the model #BUT TARGET WILL BE TARGET!! IT WONT BE CHANGED!!!! CANT BACKTRACK THEN?????
# #COMMENT / UNCOMMENT THIS PART TO TRAIN/TEST
print("Starting Training...")
node_id = 0

# episodelist, actual_rewards, cum_rewards, qsum, q_values,losslist,epsilon_list, buffer_list, random_batch_list, q_action_when_exploited, qtarlist, qvalmean, moving_sum_rew_list, Ravg,q_upd_mean = train_dqn(node_id,model, target_model,tmp_model)
print("Training Complete!")

# rewdata = np.column_stack((episodelist, actual_rewards, epsilon_list)) #combine both the lists into a single 2D array
# np.savetxt("actual_rewards.txt", rewdata, fmt="%d %.4f %.4f", header=" Episode Actual Epsilon", comments="")
# cumrewdata = np.column_stack((episodelist, cum_rewards,moving_sum_rew_list, epsilon_list)) #combine both the lists into a single 2D array
# np.savetxt("cum_rewards.txt", cumrewdata, fmt="%d %.4f %.4f %.4f", header="Episode Cumulative_Rew Moving_Avg Epsilon", comments="")
# qsumdata= np.column_stack((episodelist, qsum, qvalmean, q_upd_mean, epsilon_list)) #combine both the lists into a single 2D array
# np.savetxt("Ravg_vs_qtargetmean_VS_qsrcmean.txt", qsumdata, fmt="%d %.4f %.4f %.4f %.4f", header="Episode Qtargetmean Qsrcmean Qupdatedmean Epsilon", comments="")
# lossvalues = np.column_stack((episodelist, losslist, epsilon_list)) #combine both the lists into a single 2D array
# np.savetxt("loss.txt", lossvalues, fmt="%d %.4f %.4f", header="Episode Loss Epsilon", comments="")
# qtarlistval=np.column_stack((q_values, qtarlist))
# np.savetxt("qsrc_vs_qtarget_indiv.txt", qtarlistval,fmt="%s %s", header="qvalues qtarget",comments="")

# #Start the testing phase here  19/10/23
# #test the model for 2000 episodes
test_model = DQN(input_size, output_size) 

# modellist=[13000,14000,15000,27000,28000,29000]
modellist=[27000,28000,29000]
for modelnum in modellist:
    for runnum in range(3):
    # Load the weights into the model
        print("MODEL TESTING IS ",modelnum)
        checkpoint = torch.load(f'mw_ep_{modelnum}.pt')
        test_model.load_state_dict(checkpoint)
        # Set the model to evaluation mode
        test_model.eval()
        test_ep=3000
        trial=1
        testing_model(test_model,test_ep,modelnum,runnum)
    print("Done testing!!")


# #END OF TESTING PHASE   

