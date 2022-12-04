
import random
import time
import itertools
import sys
import numpy as np

from controller import Robot, DistanceSensor, Motor

TIME_STEP = 64
MAX_SPEED = 6.28
ACTION_TAKEN = False
FLAG=0
robot = Robot()



ps = []
psNames = ['ps0','ps1','ps2','ps3','ps4','ps5','ps6','ps7']

    
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')    
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

leftSpeed=0
rightSpeed=0

qfile = open('qlog.txt','w') 
rewardf = open('reward.txt','w') 
qsumfile = open('qsum.txt','w') 

ALPHA=0.3 #LEARNING RATE
GAMMA=0.8 #DISCOUNT RATE
EPSILON= 0.90#EXPLORATION FACTOR
#EPISODES=
NEXT_STATE=0
ACTION=0#(0:FORWARD,1:BACKWARD,2:stop,3:LEFT)
ACTION_TAKEN=False
STATES=10
STATE = 0	
REWARD=0
ACTIONS=[1,2,3,4]
NO_OF_ACTIONS=4
Q=[]
Q.append([0.0,0.0,0.0,0.0])
Q.append([0.0,0.0,0.0,0.0])
Q.append([0.0,0.0,0.0,0.0])
Q.append([0.0,0.0,0.0,0.0])
Q.append([0.0,0.0,0.0,0.0])
Q.append([0.0,0.0,0.0,0.0])
Q.append([0.0,0.0,0.0,0.0])
Q.append([0.0,0.0,0.0,0.0])
Q.append([0.0,0.0,0.0,0.0])
Q.append([0.0,0.0,0.0,0.0])

    
REWARDS= []
REWARDS.append([-10,-2,-1,10])
REWARDS.append([-10,-2,-1,10])
REWARDS.append([-10,-2,-1,10])
REWARDS.append([-10,-2,-1,10])
REWARDS.append([-10,-2,-1,10])
REWARDS.append([-10,-2,-1,10])
REWARDS.append([-10,-2,-1,10])
REWARDS.append([-10,-2,-1,10])
REWARDS.append([-10,-2,-1,10])
REWARDS.append([-10,-2,-1,10])


count=0
CREWARD=0

def forward():
    leftSpeed = 0.5*MAX_SPEED
    rightSpeed = 0.5*MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
        
def backward():
    leftSpeed = -0.5 * MAX_SPEED
    rightSpeed = -0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
        
def left():
    leftSpeed=0
    rightSpeed=0
    leftSpeed -= 0.5 * MAX_SPEED
    rightSpeed += 0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)


def right():
    leftSpeed=0
    rightSpeed=0
    leftSpeed += 0.5 * MAX_SPEED
    rightSpeed -= 0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

def stop():
    leftSpeed = 0
    rightSpeed = 0
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

def Obstacle_Avoider(): # should return True or False
        psValues = []
        for i in range(8):
        	psValues.append(ps[i].getValue())
        obstacle = psValues[0] >80.0 and psValues[7] > 80.0
        if obstacle:
        	return 1
        else:
        	return 0

def RANDOM(EXPLORATION_PARAMETER):
	RANDOM_VAR=random.randint(0, 100)
	PROB=float(RANDOM_VAR/100)
	return PROB
def DECAY(PARAMETER):
	PARAMETER=float(PARAMETER)*0.98
	return PARAMETER
def GET_STATE():
	STATE_NO=random.randint(0, 9)
	return STATE_NO

#Q_TABLE=[]  #This is not needed

def MAX(NEXT_S):   #may cause an error #removed the arg Q_TABLE
	LIST=[]
	MAX_VALUE=0.0
	b=0
	while b<=3:
		LIST.append(Q[NEXT_S][b]) #may cause an error
		b=b+1
	J=0
	while J<=2:
		if MAX_VALUE>LIST[J]:
			N1=MAX_VALUE
		else:
			N1=LIST[J]
		N2=LIST[J+1]
		DIFF= N1-N2
		if DIFF>0:
			MAX_VALUE=N1
		else:
			MAX_VALUE=N2
		J=J+1
	return MAX_VALUE

def ARGMAX(S):     #may cause an error #removed the arg Q_TABLE
	#print('S is'+str(S))
	ARRAY=[]
	MAX_VALUE=0.0
	u=0
	while u<=3:
		ARRAY.append(Q[S][u])   #may cause an error
		u=u+1
	p=0
	while p<=2:
		if MAX_VALUE>ARRAY[p]:
			N1=MAX_VALUE
		else:
			N1=ARRAY[p]
		N2=ARRAY[p+1]
		DIFF= N1-N2
		if DIFF>0:
			MAX_VALUE=N1
		else:
			MAX_VALUE=N2
		p=p+1
	r=0
	while r<=3:
		NUM=ARRAY[r]
		if NUM==MAX_VALUE:
			MAX_INDEX=r
			break
		r=r+1
	return MAX_INDEX

#removed the arg Q_TABLE
def UPDATE(S,NEXT_S,A,ACTIONS,R,LEARNING_RATE,DISCOUNT_FACTOR):
	global Q
	Q_OLD=Q[S][A]
	Q_MAX = MAX(NEXT_S)
	Q_NEW = (1-LEARNING_RATE)*Q_OLD + LEARNING_RATE*(R + DISCOUNT_FACTOR*NEXT_S)
  	#print('Q value:'+Q_NEW)
	Q[S][A]=Q_NEW
	print("THE END OF ONE ROUND !!!!!!!!!!!!!!!!!!!!!!!!")

iter_count=0

while robot.step(TIME_STEP) != -1:
	iter_count=iter_count+1
	print("Iter count is "+str(iter_count))
	for i in range(8):
		distanceSensor = robot.getDevice(psNames[i])
		#ps.append(robot.getDistanceSensor(psNames[i]))
		ps.append(distanceSensor)
		ps[i].enable(TIME_STEP)
	
	
	I=0
	y=0
	mm=100
	
	FLAG=0
	count=count+1
	while I<1:
		I=I+1
		ACTION_TAKEN = False
		#print('This is outermost')
		x=Obstacle_Avoider()
		if x==1:
			FLAG=Obstacle_Avoider()
			if(FLAG==1):
				NEXT_STATE=STATE+1
				if NEXT_STATE>=10:
					NEXT_STATE=0
				if NEXT_STATE<0:
					NEXT_STATE=0
				print('STATE:')
				print('       ' + str(STATE))
		if x==0:
			forward()
			FLAG=0
		if FLAG==1:
			PROB= RANDOM(EPSILON)
			if PROB<=EPSILON:
				ACTION=random.randint(0, 3)
				FLAG=2
			else:
				ACTION=ARGMAX(STATE)
				FLAG=2

		if FLAG==2:
			if ACTION==0:
				forward()
				#time.sleep(1.5) #subject to change
				REWARD= REWARDS[STATE][ACTION]
				
				
			if ACTION==1:
				backward()
				REWARD= REWARDS[STATE][ACTION]
				
				
			if ACTION==2:
				stop()
				REWARD= REWARDS[STATE][ACTION]
				
				
			if ACTION==3:
				left()
				REWARD= REWARDS[STATE][ACTION]
				
				
			ACTION_TAKEN=True
			print('Action is '+str(ACTION))
			time.sleep(0.5)
         
		if ACTION_TAKEN == True:
			UPDATE(STATE,NEXT_STATE,ACTION ,ACTIONS,REWARD,ALPHA ,GAMMA)
			STATE = NEXT_STATE
			EPSILON = DECAY(EPSILON)
			if EPSILON<0.5:
				EPSILON=0.9
				#print('EPISODE ENDED:')
				#print('     ' + I)
				time.sleep(7)
		CREWARD+=REWARD
		y=0
		while y<=8:
			#print('SET OF Q VALUES WILL START:')
			l=0
			while l<=3:
				#print('Q VALUE :')
				#print(Q[y][l])
				#time.sleep(2)
				l=l+1
			#time.sleep(2)
			y=y+1
	
	qsum=0
	for ql in Q:
	        qsum=qsum+sum(ql)
	qsumfile.write(str(count)+"\t"+str(qsum))
	qsumfile.write(" \n")
	qfile.write(str(Q))
	qfile.write("\n******************\n")
	rewardf.write(str(count)+"\t"+str(CREWARD))
	rewardf.write(" \n")
		

