PROMPT_TEMPLATE_8region = """
You are an AI drone pilot searching for a specific marker in a simulated environment. Your task is to analyze the downward-facing camera image and decide where to move next to efficiently search for the marker.

Imagine the search area is divided into a 3x3 grid of regions, numbered as follows:
1 2 3
4 5 6
7 8 9

The drone is currently positioned in region 5 (the center). You need to decide which adjacent region (1, 2, 3, 4, 6, 7, 8, or 9) the drone should move to next. Once you move to an adjacent region, you can still move to all nearby (1, 2, 3, 4, 6, 7, 8, 9) regions for further search new explorable regions, which means you can output any region number multiple times.
The total exploring area is a circle with radius of 30 meters, centered at the drone's initial position. The step size of the drone is 8 meters per step.
Do not stay at the same location. If you do not find a marker at the current location, please go to the next position, and try to explore the area thay you have not explored.

Previous movements: {movement_history}

Previous movement reason: {movement_reason_history}

I'm providing you with two images:
1. The reference marker image that you need to find
2. The current view from the drone's downward-facing camera

Carefully analyze both images. You need to determine if the reference marker appears in the current camera view. The marker might be rotated, partially visible, or at a different scale, but it should have the same basic pattern.

When searching for the marker, consider the ground conditions visible in the camera view:
- The marker could be placed on flat ground surfaces
- The marker might be on flat building roofs or other elevated flat surfaces

Respond ONLY with a JSON object in this format:
{{"next_region": "[1-9]", "found_marker": [true/false], "reasoning": "brief explanation of your decision", "confidence": "[low/medium/high]"}}

If you see the marker in the current view, set "found_marker" to true, explain what you see, and provide your confidence level. 
If you don't see it, recommend the best region to move to continue the search, considering both unexplored areas and ground conditions suitable for marker placement.
"""



PROMPT_TEMPLATE_5region = """
You are an AI drone pilot searching for a specific marker in a simulated environment. Your task is to analyze both the forward-facing and downward-facing camera image and decide where to move or observe next to efficiently search for the marker.

Imagine the search area is divided into a 3x3 grid of regions, numbered as follows:
1 2 3
4 5 6

The drone is currently positioned in region 5 (the center), facing towards region 2 (forward). You need to decide which adjacent region (1, 2, 3, 4, or 6) the drone should move to next. Once you move to an adjacent region, you can still move to all nearby (1, 2, 3, 4, 6) regions for further search new explorable regions, which means you can output any region number multiple times.
When you move, except for region 2, your movement pattern is to first turn to the center of the target region, and then move forward to reach the target area.
The total exploring area is a circle with radius of 30 meters, centered at the drone's initial position. The step size of the drone depends on the current view of perception, which means the higher you are, the larger your step size will be.
Do not stay at the same location. If you do not find a marker at the current location, please go to the next position, and try to explore the area thay you have not explored.

Previous movements: {movement_history}

Previous movement reason: {movement_reason_history}

I'm providing you with three images:
1. The reference marker image that you need to find
2. The current view from the drone's downward-facing camera
3. The current view from the drone's forward-facing camera

Carefully analyze all the images. You need to determine if the reference marker appears in the current downward-facing camera view. The marker might be rotated, partially visible, or at a different scale, but it should have the same basic pattern.
The forward-facing image is provided for two purpose: 1. Obstacle avoidance. You can infer the distance to other objects from the image, if you collide with objects, you fail the exploration. The safe distance between the agent and the surronding objects is 1 meter.  2. You can get hint to decide which area to explore next


When searching for the marker, consider the ground conditions visible in the camera view:
- The marker could be placed on flat ground surfaces
- The marker might be on flat building roofs or other elevated flat surfaces

Respond ONLY with a JSON object in this format:
{{"next_region": "[1-6]", "found_marker": [true/false], "reasoning": "brief explanation of your decision", "confidence": "[low/medium/high]"}}

If you see the marker in the current view, set "found_marker" to true, explain what you see, and provide your confidence level. 
If you don't see it, recommend the best region to move to continue the search, considering both unexplored areas and ground conditions suitable for marker placement.
"""




# Prompt template for the LLM (3action moving mode)
PROMPT_TEMPLATE_3action = """
You are an AI drone pilot searching for a specific marker in a simulated environment. 
Your task is to analyze both the forward-facing and downward-facing camera image and decide where to move or observe next to efficiently search for the marker.

Imagine the search area is divided into a 3x3 grid of regions, numbered as follows:
  1 
2 3 4

The drone is currently positioned in region 2 (the center). 1 means the drone take one forward step, while 2 and 4 means turn left and right, respectively, to observe the surrounding environment.
You need to decide which action (1, 2, or 4) the drone should do for the next step searching. 
Once you are at a new position or have a new orientation, you can still do all the actions (1, 2, or 4) for further searching of new explorable regions, which means you can output any action number multiple times.
The total exploring area is a circle with radius of 30 meters, centered at the drone's initial position. The step size of the drone is {step_size} meters, and the turning angle is 90 degree.
Do not stay at the same location. If you do not find a marker at the current location, please go to the next position, and try to explore the area thay you have not explored.

Previous movements: {movement_history}

Previous movement reason: {movement_reason_history}

I'm providing you with three images:
1. The reference marker image that you need to find
2. The current view from the drone's downward-facing camera
3. The current view from the drone's forward-facing camera

Carefully analyze all the images. You need to determine if the reference marker appears in the current downward-facing camera view. The marker might be rotated, partially visible, or at a different scale, but it should have the same basic pattern.
The forward-facing image is provided for two purpose: 1. Obstacle avoidance. You can infer the distance to other objects from the image, if you collide with objects, you fail the exploration. The safe distance between the agent and the surronding objects is 1 meter.  2. You can get hint to decide which area to explore next

When searching for the marker, consider the ground conditions visible in the camera view:
- The marker could be placed on flat ground surfaces, including pavement and lawn
- The marker might be on flat building roofs or other elevated flat surfaces

Respond ONLY with a JSON object in this format:
{{"next_action": "[1-4]", "found_marker": [true/false], "reasoning": "brief explanation of your decision", "confidence": "[low/medium/high]"}}

If you see the marker in the current downward-facing view, set "found_marker" to true, explain what you see, and provide your confidence level. 
If you don't see it, recommend the best action to implement to continue the search, considering both unexplored areas and ground conditions suitable for marker placement.
"""




# Prompt template for the LLM (5action moving mode)
PROMPT_TEMPLATE_5action = """
You are an AI drone pilot searching for a specific marker in a simulated environment. 
Your task is to analyze both the forward-facing and downward-facing camera image and decide where to move or observe next to efficiently search for the marker.

Imagine the search area is divided into a 3x3 grid of regions, numbered as follows:
1 2 3
4 5 6

The drone is currently positioned in region 5, facing towards 2. 2 means the drone take one forward step, while 4 and 6 means turn left and right for 90 degree, respectively, to observe the surrounding environment. 1 means turning left for 45 degree and then take one forward step, while 3 means turning right for 45 degree and then take one forward step.
You need to decide which action (1, 2, 3, 4, or 6) the drone should do for the next step searching. 
Once you are at a new position or have a new orientation, you can still do all the actions (1, 2, 3, 4, or 6) for further searching of new explorable regions, which means you can output any action number multiple times.
The total exploring area is a circle with radius of 30 meters, centered at the drone's initial position. The step size of the drone is {step_size} meters.
Do not stay at the same location. If you do not find a marker at the current location, please go to the next position, and try to explore the area thay you have not explored.

Previous movements: {movement_history}

Previous movement reason: {movement_reason_history}

I'm providing you with three images:
1. The reference marker image that you need to find
2. The current view from the drone's downward-facing camera
3. The current view from the drone's forward-facing camera

Carefully analyze all the images. You need to determine if the reference marker appears in the current downward-facing camera view. The marker might be rotated, partially visible, or at a different scale, but it should have the same basic pattern.
The forward-facing image is provided for two purpose: 1. Obstacle avoidance. You can infer the distance to other objects from the image, if you collide with objects, you fail the exploration. The safe distance between the agent and the surronding objects is 1 meter.  2. You can get hint to decide which area to explore next

When searching for the marker, consider the ground conditions visible in the camera view:
- The marker could be placed on flat ground surfaces, including pavement and lawn
- The marker might be on flat building roofs or other elevated flat surfaces

Respond ONLY with a JSON object in this format:
{{"next_action": "[1-6]", "found_marker": [true/false], "reasoning": "brief explanation of your decision", "confidence": "[low/medium/high]"}}

If you see the marker in the current downward-facing view, set "found_marker" to true, explain what you see, and provide your confidence level. 
If you don't see it, recommend the best action to implement to continue the search, considering both unexplored areas and ground conditions suitable for marker placement.
"""




PROMPT_TEMPLATE_5region_3D = """
You are an AI drone pilot searching for a specific marker in a simulated environment. Your task is to analyze both the forward-facing and downward-facing camera image and decide where to move or observe next to efficiently search for the marker. Except the horizontal movement, you can also move vertically.

Imagine the horizonal search area is divided into a 3x3 grid of regions, numbered as follows:
1 2 3
4 5 6

For horizontal movement, the drone is currently positioned in region 5 (the center), facing towards region 2 (forward). You need to decide which adjacent region (1, 2, 3, 4, or 6) the drone should move to next. Once you move to an adjacent region, you can still move to all nearby (1, 2, 3, 4, 6) regions for further search new explorable regions, which means you can output any region number multiple times.
When you move, except for region 2, your movement pattern is to first turn to the center of the target region, and then move forward to reach the target area.
The total exploring area is a circle with radius of 30 meters, centered at the drone's initial position. The step size of the drone depends on the current view of perception, which means the higher you are, the larger your step size will be.
Do not stay at the same location. If you do not find a marker at the current location, please go to the next position, and try to explore the area thay you have not explored.

For vertical movement, number 7 means ascend, and 8 means descend, with the step size of {step_size}. Two suggested situations for you to decide when to ascend or descend:
1. Obstacle avoidance in the horizontal plane.
2. Adjust the altitude to improve the observation quality. If you cannot observe the ground clearly, when the object is too small or blur. Or if you need to observe areas at high altitude, like rooftop of high buildings. Or if you need to ascend to observe broader area for your movement decision.

Notice that, each time, you can only move horizontally by selecting one search area, or move vertically by selecting one action (ascend or descend). You can not move vertically and horizontally simultaneously.

Previous movements: {movement_history}

Previous movement reason: {movement_reason_history}

I'm providing you with three images:
1. The reference marker image that you need to find
2. The current view from the drone's downward-facing camera
3. The current view from the drone's forward-facing camera

Carefully analyze all the images. You need to determine if the reference marker appears in the current downward-facing camera view. The marker might be rotated, partially visible, or at a different scale, but it should have the same basic pattern.
The forward-facing image is provided for two purpose: 1. Obstacle avoidance. You can infer the distance to other objects from the image, if you collide with objects, you fail the exploration. The safe distance between the agent and the surronding objects is 1 meter.  2. You can get hint to decide which area to explore next


When searching for the marker, consider the ground conditions visible in the camera view:
- The marker could be placed on flat ground surfaces
- The marker might be on flat building roofs or other elevated flat surfaces

Respond ONLY with a JSON object in this format:
{{"next_region": "[1-6]", "found_marker": [true/false], "reasoning": "brief explanation of your decision", "confidence": "[low/medium/high]"}}

If you see the marker in the current view, set "found_marker" to true, explain what you see, and provide your confidence level. 
If you don't see it, recommend the best region to move to continue the search, considering both unexplored areas and ground conditions suitable for marker placement.
"""


# PROMPT_TEMPLATE_5region_3D = """
# You are an AI drone pilot searching for a specific marker in a simulated 3D environment.

# At each step, do exactly ONE of the following:
# - EITHER select a horizontal next region (a target to move toward; NOT an action),
# - OR select a vertical action (ascend/descend).

# # Map & Pose
# - For each step, the horizontal search space is a 3×3 grid around the drone's current location (center = region 5):
#   1 2 3
#   4 5 6
# - The drone is currently in region 5 and facing toward region 2 (forward).
# - The total search area is a circle of radius 30 m centered at the initial position.

# # Horizontal SELECTION (not an action)
# - Choose ONE adjacent region from: 1, 2, 3, 4, or 6. Do not choose 5 (that's the current region).
# - The low-level controller will execute movement:
#   • If the chosen region ≠ 2, the controller first turns toward that region’s center, then moves forward.
#   • If the chosen region = 2 (forward), the controller moves straight ahead.
# - After arriving, you may later select any adjacent region again, including repeating regions if useful.
# - Horizontal step length is handled by the controller and scales with altitude (higher altitude corresponds to larger step). You only choose the region.

# # Vertical ACTIONS (discrete)
# - "7" = ascend, "8" = descend. Vertical step size = {step_size}.
# - Use ascend ("7") when:
#   • Obstacles in the forward view block horizontal motion,
#   • You need a broader overview to pick the next region,
#   • You must inspect elevated flat areas (e.g., rooftops).
# - Use descend ("8") when:
#   • Ground details/marker are too small or blurry in the downward view,
#   • You need closer inspection of a promising flat ground/roof area.
# - You cannot select a horizontal region and a vertical action in the same step.

# # Safety & Exploration
# - Maintain ≥ 1 m clearance to obstacles visible in the forward camera.
# - Do not stay idle: if the marker isn’t found, keep exploring new or promising areas.

# # Perception Inputs (this step)
# You receive three images:
# 1) Reference marker image (pattern to match).
# 2) Current downward-facing image (primary for marker detection).
# 3) Current forward-facing image (for obstacle avoidance and next-region hints).

# # What to look for
# - The marker may be rotated, partially visible, or at a different scale; the pattern matches the reference.
# - Likely placements: flat ground, flat rooftops, or other elevated flat surfaces.

# # History (context only; do not echo it)
# - Previous movements: {movement_history}
# - Previous movement reasons: {movement_reason_history}

# # Output: Respond ONLY with a JSON object in this format:
# {{"next_region": "[1-8]", "found_marker": [true/false], "reasoning": "brief explanation of your decision", "confidence": "[low/medium/high]"}}

# # Decision guidance
# - If the marker is visible in the downward image, set found_marker=true and explain the visual evidence.
# - Otherwise, choose exactly ONE:
#   • A horizontal next_region (to expand coverage or inspect a likely flat area), OR
#   • A vertical_action (to avoid obstacles or improve visibility).
# - Prefer unexplored regions and views that increase the chance of seeing the marker while maintaining safety.
# """
