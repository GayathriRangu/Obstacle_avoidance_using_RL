from controller import Supervisor
import random
import math, time

# --- Constants ---
TIME_STEP = 64
MAX_SPEED = 6.28
TARGET_THRESHOLD = 0.2
STUCK_THRESHOLD = 0.01
STUCK_STEPS = 15

# --- Arena bounds (based on your corners) ---
X_MIN, X_MAX = -0.42, 0.42
Z_MIN, Z_MAX = -0.42, 0.42
Y_POS = 0.035  # e-puck height
# --- Robot & Devices ---
robot = Supervisor()
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

ps_names = ['ps0','ps1','ps2','ps3','ps4','ps5','ps6','ps7']
ps = []
for name in ps_names:
    sensor = robot.getDevice(name)
    sensor.enable(TIME_STEP)
    ps.append(sensor)

# --- Target: Wooden Box ---
target_box = robot.getFromDef("TARGET_BOX")
target_color_field = None
if target_box:
    # Navigate to the box’s Shape → Appearance → baseColor
    try:
        shape = target_box.getField("children").getMFNode(0)
        appearance = shape.getField("appearance").getSFNode()
        target_color_field = appearance.getField("baseColor")
    except:
        print("⚠️ Warning: Could not access target box appearance fields")

# --- Helper Functions ---
def distance(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_bearing(current, target):
    dx = target[0] - current[0]
    dz = target[1] - current[1]
    return math.atan2(dz, dx)

def random_position_within_arena():
    """Generate a safe random spawn position inside arena bounds"""
    x = random.uniform(X_MIN, X_MAX)
    z = random.uniform(Z_MIN, Z_MAX)
    return [x, Y_POS, z]

def clamp_speed(speed):
    return max(-MAX_SPEED, min(MAX_SPEED, speed))

def set_speed(left, right):
    left_motor.setVelocity(clamp_speed(left))
    right_motor.setVelocity(clamp_speed(right))

# --- Initialization ---
goal_pos = [-0.0216616, 0.450585, 0.0488352]  # Fixed target position
print("Fixed target:", goal_pos)

prev_pos = gps.getValues()
print("Starting position:", prev_pos)
stuck_counter = 0

# --- Main Loop ---
while robot.step(TIME_STEP) != -1:
    current_pos = gps.getValues()
    print("Current position:", current_pos)
    dist_to_target = distance(current_pos, goal_pos)
    print("Distance to target:", dist_to_target)
    bearing = get_bearing(current_pos, goal_pos)
    # dx = current_pos[0] - prev_pos[0]
    # dy = current_pos[1] - prev_pos[1]
    # dz = current_pos[2] - prev_pos[2]
    # print(f"ΔX: {dx:.5f}, ΔY: {dy:.5f}, ΔZ: {dz:.5f}")
    # prev_pos = current_pos
    # --- Obstacle Avoidance ---
    ps_values = [s.getValue() for s in ps]
    front = max(ps_values[0], ps_values[1], ps_values[2], ps_values[3], ps_values[4])
    left = max(ps_values[5], ps_values[6], ps_values[7])
    right = max(ps_values[0], ps_values[1], ps_values[2])

    left_speed = MAX_SPEED
    right_speed = MAX_SPEED

    if front > 80:
        left_speed = -0.5 * MAX_SPEED
        right_speed = 0.5 * MAX_SPEED
    elif left > 80 and right > 80:
        # Surrounded → back up
        left_speed = -0.5 * MAX_SPEED
        right_speed = -0.5 * MAX_SPEED
    elif left > 80:
        left_speed = 0.5 * MAX_SPEED
        right_speed = -0.5 * MAX_SPEED
    elif right > 80:
        left_speed = -0.5 * MAX_SPEED
        right_speed = 0.5 * MAX_SPEED

    # --- Target Following ---
    K = 1.5  # steering gain
    left_speed += K * bearing
    right_speed -= K * bearing

    # --- Stuck Detection ---
    if distance(current_pos, prev_pos) < STUCK_THRESHOLD:
        stuck_counter += 1
    else:
        stuck_counter = 0
    prev_pos = current_pos

    if stuck_counter > STUCK_STEPS:
        print("Stuck detected, rotating to escape...")
        left_speed = MAX_SPEED * random.choice([-1, 1])
        right_speed = -left_speed
        stuck_counter = 0

    # --- Target Reached ---
    if dist_to_target < TARGET_THRESHOLD:
        print("###################################################################################################Reached target:", goal_pos)
        # Change box color to green
        if target_color_field:
            target_color_field.setSFColor([0, 1, 0])

        # Respawn robot safely inside arena
        new_pos = random_position_within_arena()
        robot.getSelf().getField("translation").setSFVec3f(new_pos)
        robot.getSelf().getField("rotation").setSFRotation([0, 1, 0, 0])  # No tilt
        # time.sleep(0.1)  # Allow position update
        robot.step(TIME_STEP)
        # Keep it green for ~1s, then revert
        for _ in range(15):  # 15 * 64ms ≈ 1 second
            robot.step(TIME_STEP)
        if target_color_field:
            target_color_field.setSFColor([0.6, 0.4, 0.2])  # brown again
        continue

    # --- Apply Speeds ---
    set_speed(left_speed, right_speed)
    # time.sleep(0.1)
