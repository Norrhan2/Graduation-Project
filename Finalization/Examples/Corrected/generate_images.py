"""
Tariikhna - Generate Images for Two Sets (FLUX dev) - VS Code / Local version

Reads story JSONs from an input folder and routes them into two output folders:
  output_base/  <- files WITHOUT _1   (e.g. abu_bakr_frees_the_slaves_corrected.json)
  output_v1/    <- files WITH _1      (e.g. abu_bakr_frees_the_slaves_corrected_1.json)

Each output folder gets:
  images/   the generated panel PNGs
  stories/  the JSONs with an added image_file field (ready for Streamlit)

------------------------------------------------------------------
SETUP (run these in the VS Code terminal once):
    pip install fal-client requests
Then set your FAL.ai key (get one at https://fal.ai/dashboard/keys):
    Windows PowerShell:   $env:FAL_KEY="your-key-here"
    macOS/Linux:          export FAL_KEY="your-key-here"
  (or just paste it into FAL_KEY below)

USAGE:
    1. Put all your story JSONs in the folder set by INPUT_DIR below.
    2. Run:  python generate_images.py
------------------------------------------------------------------
"""

import os
import json
import time
import glob
import requests
import fal_client

# ============================================================
# CONFIGURATION - edit these
# ============================================================

# Your FAL.ai API key. Better to set it as an environment variable (see SETUP above),
# but you can also paste it here directly.
FAL_KEY = os.environ.get("FAL_KEY", "0de17aeb-1e1b-42a7-9c78-c9f0d2ee6b51:c30735f7b3792f63391ae672b69dc3d2")
os.environ["FAL_KEY"] = FAL_KEY

# Folder containing your input JSON files (all 8 of them).
# '.' means the same folder as this script. Change if your JSONs are elsewhere,
# e.g. INPUT_DIR = r"C:\Users\moham\Desktop\tariikhna\stories"
INPUT_DIR = r"C:\Users\Norhan Yasser\Graduation-Project\Finalization\Examples\Corrected\input_stories"

# Where to write the two output sets
BASE_OUT = "output_base"
# V1_OUT = "output_v1"

MODEL = "fal-ai/nano-banana"      # FLUX dev as requested  // changed to nano-banana 
IMAGE_SIZE = "landscape_4_3"   # comic panel shape

# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(prompt, out_path):
    """Generate one image via FLUX dev, save to out_path. Returns True on success."""
    try:
        result = fal_client.subscribe(
            MODEL,
            arguments={
                "prompt": prompt,
                "image_size": IMAGE_SIZE,
                "num_images": 1,
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
            },
        )
        images = result.get("images", [])
        if not images:
            print("    No image returned (possibly filtered)")
            return False
        url = images[0]["url"]
        img_data = requests.get(url, timeout=60).content
        with open(out_path, "wb") as f:
            f.write(img_data)
        return True
    except Exception as e:
        print(f"    ERROR: {str(e)[:120]}")
        return False


# ============================================================
# FILE ROUTING
# ============================================================

def is_v1(fname):
    """A file is a '_1' file if its name (before .json) ends with _1."""
    stem = fname[:-5] if fname.endswith(".json") else fname
    return stem.endswith("_1")


def process_set(file_list, out_dir, label):
    """Generate images for every story in file_list into out_dir."""
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "stories"), exist_ok=True)

    for fpath in sorted(file_list):
        fname = os.path.basename(fpath)
        with open(fpath, encoding="utf-8") as f:
            story = json.load(f)

        pid = fname[:-5]  # strip .json, keep full name including _1 if present
        title = story.get("story_context", {}).get("story_title", pid)
        panels = story.get("panels", [])
        print(f"\n[{label}] {title}  ({len(panels)} panels)")

        for panel in panels:
            n = panel["panel_number"]
            img_name = f"{pid}_panel_{n}.png"
            img_path = os.path.join(out_dir, "images", img_name)
            print(f"  Panel {n}...", end=" ", flush=True)
            ok = generate_image(panel.get("image_prompt", ""), img_path)
            panel["image_file"] = img_name if ok else None
            print("OK" if ok else "FAILED")
            time.sleep(1)

        out_json = os.path.join(out_dir, "stories", f"{pid}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(story, f, ensure_ascii=False, indent=2)

    print(f"\n[{label}] complete -> {out_dir}/")


# ============================================================
# MAIN
# ============================================================

def main():
    if FAL_KEY == "your-fal-api-key-here":
        print("ERROR: set your FAL.ai API key first (see SETUP at the top of this file).")
        return

    # Find all JSON files in the input folder
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if not all_files:
        print(f"No JSON files found in '{INPUT_DIR}/'.")
        print("Put your story JSONs there, or change INPUT_DIR at the top of this file.")
        return

    base_files = [f for f in all_files if not is_v1(os.path.basename(f))]
    # v1_files = [f for f in all_files if is_v1(os.path.basename(f))]

    print(f"Found {len(all_files)} JSON files in '{INPUT_DIR}/'")
    print(f"\nBASE set ({len(base_files)}) -> {BASE_OUT}/")
    for f in sorted(base_files):
        print(f"  {os.path.basename(f)}")
    # print(f"\nV1 set ({len(v1_files)}) -> {V1_OUT}/")
    #for f in sorted(v1_files):
     #   print(f"  {os.path.basename(f)}")

    if base_files:
        process_set(base_files, BASE_OUT, "BASE")
    #if v1_files:
     #   process_set(v1_files, V1_OUT, "V1")

    print("\n\nAll done!")
    print(f"Images and updated JSONs are in '{BASE_OUT}/'.")


if __name__ == "__main__":
    main()
