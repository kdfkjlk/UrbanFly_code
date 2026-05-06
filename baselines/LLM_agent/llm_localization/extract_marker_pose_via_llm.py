import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)



import numpy as np
import cv2
import json
import base64
from openai import OpenAI
from io import BytesIO
from PIL import Image
import time
import math
import cairosvg
import os
from natsort import natsorted


from configs.api_config import *


openai_client = OpenAI(
    api_key=api_key
)


# Function to encode image to base64 for API
def encode_image(image_array):
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    # Save to BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    # Get the byte data and encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def encode_svg_marker(svg_path):
    # Create a temporary PNG file
    temp_png = "temp_marker.png"
    
    # Convert SVG to PNG using cairosvg
    cairosvg.svg2png(url=svg_path, write_to=temp_png)
    
    # Read the PNG and encode to base64
    with open(temp_png, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Clean up the temporary file
    if os.path.exists(temp_png):
        os.remove(temp_png)
        
    return encoded_string



def quiry_llm(prompt, encoded_image, encoded_marker):
    try:
    
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_marker}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=512,
            timeout=60
        )
            
        # Parse the response
        response_text = response.choices[0].message.content
        # Extract JSON from response (in case there's additional text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        result = json.loads(json_str)
        print(f"Result: {result}")
                          
    except Exception as e:
        print("=" * 80)
        print("Error type:", type(e).__name__)
        print("Error message:", repr(e))
        if "response" in locals() and response is not None:
            print("Raw response:", response)
        print("=" * 80)
        result = None

    return result
    


PROMPT_TEMPLATE_extract_marker_pose = """
You are an AI assistant specialized in visual marker detection and localization. Your task is to find a specific ArUco-style marker in a drone camera image.

I'm providing you with two images:
1. The reference marker image (first image) - this is the exact marker you need to find
2. The current view from the drone's downward-facing camera (second image) - search for the marker in this image

Instructions:
- Carefully examine the drone camera image to locate the reference marker
- The marker may appear at different scales, orientations, or lighting conditions
- If you find the marker, determine its center position in pixel coordinates
- The coordinate system origin (0,0) is at the top-left corner of the image. The x-axis extends to the right, corresponding to the image width, while the y-axis extends downward, corresponding to the image height.
- Image dimensions: width={Image_width} pixels, height={Image_height} pixels

Output Format:
If marker is found, respond with JSON:
{{"marker_position": [x, y], "confidence": "high/medium/low", "found": true}}

If marker is NOT found, respond with JSON:
{{"marker_position": null, "confidence": null, "found": false}}

Be precise with the center coordinates and honest about detection confidence.
"""





def extract_marker_pose_one_image(data_dir, file_name, encoded_marker, map_name=None, move_type=None):

    explore_result_path = os.path.join(data_dir, file_name, "explore.json")
    with open(explore_result_path, 'r') as f:
            explore_result = json.load(f)

    # if found marker
    if explore_result[-1]['found_marker']:
        # Find the image_down_i.jpg with the largest i
        folder_path = os.path.join(data_dir, file_name)
        image_files = [f for f in os.listdir(folder_path) if f.startswith('image_down_') and f.endswith('.jpg')]
            
        # Sort by number and get the largest
        image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        image_path = os.path.join(folder_path, image_files[0])
                
        image = cv2.imread(image_path)
        encoded_image = encode_image(image)

        # Fix: use correct parameter names
        prompt = PROMPT_TEMPLATE_extract_marker_pose.format(
            Image_width=image.shape[1], 
            Image_height=image.shape[0]
        )

        response = quiry_llm(prompt, encoded_image, encoded_marker)

        if response is not None and response.get('found', False):
            print(f"Marker found! Position: {response['marker_position']}, Confidence: {response.get('confidence', 'unknown')}")
        else:
            print("Marker not found")

        # Store result with file identifier
        result_entry = {
            'file_name': file_name,
            'result': response,
            'image_width': image.shape[1],
            'image_height': image.shape[0]
        }

        # ## save the prediction as image
        # bbox, center = get_coord_from_request(result_entry, format='center')

        # if map_name is not None and move_type is not None:
        #     save_path=f'./check_saved_marker_pose_center_{map_name}_{move_type}/{file_name}_label.png'
        #     draw_bbox_and_center_on_array(image, bbox, center, save_path=save_path)


    else:
        print(f"Skipping {file_name} - no marker found in exploration")
        result_entry = {
            'file_name': file_name,
            'result': None
        }
    
    time.sleep(2)
    return result_entry
    


def main(
    data_dir,
    result_save_path,
    marker_path=None,
    map_name=None,
    move_type=None
):
    """
    Extract marker center pixel coordinates from the final downward image
    of each episode generated by discovery_llm.py.

    Args:
        data_dir: Directory containing episode folders, each with explore.json and image_down_*.jpg.
        result_save_path: Path to save LLM marker localization results.
        marker_path: Path to marker.svg. If None, use PROJECT_ROOT/llm_nav/marker.svg.
        map_name: Optional map name, only used for visualization/debug naming.
        move_type: Optional move type, e.g., 2D or 3D.
    """

    if marker_path is None:
        marker_path = os.path.join(PROJECT_ROOT, 'llm_nav', 'marker.svg')

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not os.path.isfile(marker_path):
        raise FileNotFoundError(f"Marker file not found: {marker_path}")

    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

    sample_files = [
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    ]
    sample_files = natsorted(sample_files)

    all_results = {}

    encoded_marker = encode_svg_marker(marker_path)

    for file_name in sample_files:
        print(f"Processing file: {file_name}")

        result_entry = extract_marker_pose_one_image(
            data_dir,
            file_name,
            encoded_marker,
            map_name=map_name,
            move_type=move_type
        )

        all_results[str(file_name)] = result_entry

    with open(result_save_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {result_save_path}")





def load_result(result_path):
    with open(result_path, "r") as f:
        result = json.load(f)
    return result





if __name__ == "__main__":

    map_name = 'ModernCityEnvironment'
    move_type = '2D'
    move_format = '5patch'

    data_dir = os.path.join(
        PROJECT_ROOT,
        'logs',
        f'Nav_{move_format}_{map_name}'
    )

    result_save_path = os.path.join(
        PROJECT_ROOT,
        'logs',
        f'Localize_{map_name}_{move_type}.json'
    )

    main(
        data_dir=data_dir,
        result_save_path=result_save_path,
        map_name=map_name,
        move_type=move_type
    )