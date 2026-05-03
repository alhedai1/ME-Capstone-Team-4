# """Label all images in data/extracted_frames/phone1 in batches and save results.

# This script will:
#  - collect all JPGs from data/extracted_frames/phone1
#  - send them to the Roboflow workflow in configurable batches
#  - append each batch's result as a JSON line to roboflow/results_phone1.jsonl

# Adjust BATCH_SIZE below if needed.
# """

# import os
# import glob
# import json
# import time
# from typing import List

# # 1. Import the library (helpful error if missing)
# try:
#     from inference_sdk import InferenceHTTPClient
# except Exception as e:
#     raise SystemExit(
#         "inference_sdk is not installed in the active environment.\n"
#         "Activate the virtualenv and install the SDK, e.g.:\n"
#         "  source ../new_env/bin/activate\n"
#         "  pip install inference-sdk\n"
#         f"(import error: {e})"
#     )

# # 2. Connect to your workflow
# client = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="5In3YOC6vN2MykmLnJV3"
# )


# def chunk_list(lst: List[str], size: int) -> List[List[str]]:
#     return [lst[i : i + size] for i in range(0, len(lst), size)]


# def main(batch_size: int = 10, pause_seconds: float = 1.0):
#     # Build a list of all jpg images in data/extracted_frames/phone1 (relative to this script)
#     script_dir = os.path.dirname(__file__)
#     phone1_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'extracted_frames', 'phone1'))
#     image_paths = sorted(glob.glob(os.path.join(phone1_dir, '*.jpg')))

#     if not image_paths:
#         print(f"No images found in {phone1_dir}")
#         return

#     out_file = os.path.join(script_dir, 'results_phone1.jsonl')
#     batches = chunk_list(image_paths, batch_size)
#     print(f"Found {len(image_paths)} images, sending in {len(batches)} batches (batch_size={batch_size})")

#     # Open output file for appending results as JSON lines
#     with open(out_file, 'a', encoding='utf-8') as fh:
#         for idx, batch in enumerate(batches, start=1):
#             print(f"Sending batch {idx}/{len(batches)} ({len(batch)} images)")
#             try:
#                 result = client.run_workflow(
#                     workspace_name="sth-mswrs",
#                     workflow_id="find-bells-and-poles-2",
#                     images={"image": batch},
#                     use_cache=True,
#                 )
#             except Exception as e:
#                 # Record the error for this batch and continue
#                 print(f"Error on batch {idx}: {e}")
#                 record = {
#                     'batch_index': idx,
#                     'images': batch,
#                     'error': str(e),
#                 }
#                 fh.write(json.dumps(record) + "\n")
#             else:
#                 # Save the successful response (include the input paths for traceability)
#                 record = {
#                     'batch_index': idx,
#                     'images': batch,
#                     'result': result,
#                 }
#                 fh.write(json.dumps(record) + "\n")

#             # polite pause between requests
#             time.sleep(pause_seconds)

#     print(f"Finished. Results appended to {out_file}")


# if __name__ == '__main__':
#     # Default batch size is 10; change if you want smaller/larger batches
#     main(batch_size=10, pause_seconds=1.0)


# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="5In3YOC6vN2MykmLnJV3"
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="sth-mswrs",
    workflow_id="find-bells-and-poles-2",
    images={
        "image": "/home/ahmed/Other/capstone/data/extracted_frames/phone1/20260403_172554_00000.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)
