import argparse
import json
import os
import pickle

import tiktoken
from nuscenes.nuscenes import NuScenes
from prompt_message import generate_assistant_message, generate_user_message


os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/anvme/workspace/b305bb10-zyt/FSDrive/.cache/tiktoken")

system = "You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters. Based on the provided particulars, please output the latent ego-path-interacting surrounding vehicle intentions, the CAM_FRONT image at the 0.5 second in the future, and plan waypoints (0.5s intervals) for the next 3 seconds."


def format_visual_tokens(indices):
    indices = str(indices).replace(" ", "")
    numbers = indices.strip("[]").split(",")
    return "".join([f"<|{num}|>" for num in numbers])


def format_agent_intent_token(intent_id):
    return f"<|agent_intent_{int(intent_id):03d}|>"


def format_agent_intent_tokens(entries, top_k):
    tokens = []
    for entry in entries[:top_k]:
        tokens.append(format_agent_intent_token(entry["intent"]))

    while len(tokens) < top_k:
        tokens.append("<|agent_none|>")

    return "".join(tokens)


parser = argparse.ArgumentParser(description="Build SFT data with ego-path interaction agent intention supervision.")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
parser.add_argument(
    "--agent_intent_labels",
    type=str,
    default="./create_data/agent_interaction_intent_labels_k64_top4.json",
)
args = parser.parse_args()

data = pickle.load(open("./create_data/cached_nuscenes_info.pkl", "rb"))
split = json.load(open("./create_data/full_split.json", "r"))
agent_intent_payload = json.load(open(args.agent_intent_labels, "r"))
agent_intent_labels = agent_intent_payload["labels"][args.split]
top_k = int(agent_intent_payload["metadata"]["top_k"])
tokens = split[args.split]

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_language_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0
traj_only = True

dataroot = "./LlamaFactory/data/nuscenes"
nusc = NuScenes(version="v1.0-trainval", dataroot=dataroot, verbose=True)
sft_indices = json.load(open("./MoVQGAN/gt_indices_sft.json"))
train_messages = []

for token in tokens:
    assitant_message = generate_assistant_message(data, token, traj_only=traj_only)
    user_message, images_path = generate_user_message(data, token)

    num_language_tokens += len(encoding.encode(user_message))
    num_user_tokens += len(encoding.encode(user_message))
    num_language_tokens += len(encoding.encode(assitant_message))
    num_assistant_tokens += len(encoding.encode(assitant_message))

    try:
        next_token = nusc.get("sample", token)["next"]
        next_img_token = format_visual_tokens(sft_indices[next_token]["CAM_FRONT"])
    except Exception:
        continue

    agent_intent_token = format_agent_intent_tokens(agent_intent_labels.get(token, []), top_k)

    train_message = {
        "id": token,
        "images": images_path,
        "system": system,
        "conversations": [
            {
                "from": "human",
                "value": "Here are current six images from the car: 'CAM_FRONT': <image>\n,'CAM_FRONT_LEFT': <image>\n, 'CAM_FRONT_RIGHT': <image>\n,'CAM_BACK': <image>\n,'CAM_BACK_LEFT': <image>\n, 'CAM_BACK_RIGHT': <image>\n"
                + user_message
                + "Based on the provided particulars, please output the latent ego-path-interacting surrounding vehicle intentions, the CAM_FRONT image at the 0.5 second in the future, and plan waypoints (0.5s intervals) for the next 3 seconds.\n",
            },
            {
                "from": "gpt",
                "value": agent_intent_token
                + " These are the latent ego-path-interacting surrounding vehicle intentions. \n"
                + next_img_token
                + " These are the visual tokens of CAM_FRONT image at the 0.5 second in the future. \n"
                + assitant_message
                + " These are the future waypoints. \n <|endoftext|><|im_end|>",
            },
        ],
    }
    train_messages.append(train_message)

print("#### Cost Summarization ####")
print(f"Number of user tokens: {num_user_tokens}")
print(f"Number of assistant tokens: {num_assistant_tokens}")
print(f"Number of total tokens: {num_language_tokens}")
print(f"Number of samples: {len(train_messages)}")

with open(f"./LlamaFactory/data/{args.split}_cot_motion_agent_interaction_intent.json", "w") as f:
    json.dump(train_messages, f, indent=4)
