from ast import literal_eval

import torch
from transformers import pipeline
import json
import os
import ast

try:
    from transformers.utils import is_flash_attn_2_available
except ImportError:
    def is_flash_attn_2_available():
        return False  # Fallback if the function is not available

# Check if CUDA is available
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Please ensure your GPU is properly set up.")

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

# Define the device index (ensure it's within the range of available GPUs)
device_index = 0  # Change this based on your system configuration
if device_index >= num_gpus:
    raise ValueError(f"Specified device cuda:{device_index} is not available. Available devices: {num_gpus}")

device = f"cuda:{device_index}"

# Initialize the pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",  # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device=device,  # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
)

# Define the audio file path

audio_file_path = input("Enter file path of the audio you want to be transcribed")
audio_file_path = audio_file_path.replace('"', '') # clean if user added "" marks

# Check if the audio file exists
if not os.path.isfile(audio_file_path):
    raise FileNotFoundError(f"The specified audio file does not exist: {audio_file_path}")

# Run the pipeline
outputs = pipe(
    audio_file_path,
    chunk_length_s=30,
    batch_size=24,
    return_timestamps=True,
)

# Print the outputs
# print(outputs)

# Define the output directory and file name
output_dir = "output_dir"
output_file_name = "outputs.txt"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the full path to the output file
output_file_path = os.path.join(output_dir, output_file_name)

# Save the outputs to a text file
with open(output_file_path, "w", encoding="utf-8") as f:
    f.write(str(outputs))

print(f"Outputs have been saved to: {output_file_path}")
