"""
Tariikhna Batch Schema Generator
==================================
Converts all curated Sira passages into v2 JSON panel schemas using an LLM API.

Features:
- Supports Groq, Together AI, Google Gemini, OpenAI
- Auto-saves progress every 10 passages
- Can resume from where it left off if interrupted
- Validates each output before saving
- Logs failures for retry

SETUP:
    pip install groq together google-generativeai openai

USAGE:
    1. Set your API key and provider below
    2. Place curated_passages_287.json in the same directory
    3. Run: python generate_training_data.py
    4. Output: training_dataset/ folder with individual JSONs + combined dataset

RECOMMENDED PROVIDERS (by cost):
    1. Gemini 1.5 Flash - free tier 1500 req/day (best for this)
    2. Groq - free tier (fast but rate limited)
    3. Together AI - $5 free credits
    4. OpenAI GPT-4o - ~$2-3 for 287 passages
"""

import json
import time
import os
import sys
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
PROVIDER = "google"  # Options: "groq", "together", "google", "openai"
API_KEY = "your-api-key-here"
MODEL_ID = {
    "groq": "llama-3.3-70b-versatile",
    "together": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "google": "gemini-1.5-flash",
    "openai": "gpt-4o",
}

INPUT_FILE = "curated_passages_287.json"
OUTPUT_DIR = "training_dataset"
PROGRESS_FILE = "generation_progress.json"

# Rate limiting (seconds between calls)
DELAY = {
    "groq": 3,      # Free tier is rate-limited
    "together": 2,
    "google": 1,     # Gemini free tier is generous
    "openai": 1,
}

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """You are Tariikhna, an AI that converts Islamic historical narratives (Sira) into structured comic panel descriptions for children aged 6-12.

Given a passage, output ONE JSON object describing ONE comic panel.

## ISLAMIC DEPICTION RULES (CRITICAL)
- Prophets (Muhammad, Ibrahim, Musa, Isa, Ismail, ALL prophets): NEVER show face. depiction_rule must be NO_FACE, FROM_BEHIND, or SILHOUETTE.
- Angels: represent as light or voice only, never as a humanoid figure.
- Sahaba (Abu Bakr, Umar, Ali, Uthman, etc.): faces allowed, depiction_rule = FULL.
- Women: modest clothing (hijab/khimar covering hair), hands and face may be visible.
- No graphic violence — battles shown through dust clouds, distant silhouettes, or aftermath.
- No idols shown in detail.
- No modern objects or anachronisms. All items must be historically accurate.
- Content must be appropriate and non-frightening for children aged 6-12.

## CHARACTER APPEARANCE
For each character, write a detailed one-paragraph physical description including: approximate age, skin tone (historically accurate for the region), build, facial hair, specific clothing with colors, headwear, footwear, and any distinguishing features. This description must be detailed enough that an image generation model can reproduce the same character consistently.

## HISTORICAL ACCURACY
- 7th century Arabian Peninsula: thobes, bisht (cloaks), imamah (turbans), izaar, leather sandals
- Architecture: mud-brick and stone, flat roofs, narrow alleys
- Lighting: oil lamps, torches, sunlight, moonlight only
- Objects: clay pots, leather water skins, woven mats, wooden items
- For ancient prophets (Ibrahim, Musa): adjust to the appropriate earlier era

## OUTPUT FORMAT
Respond with ONLY valid JSON. No markdown code fences. No explanation before or after.

{
  "scene_title": "short descriptive title",
  "era": "one of: pre_islamic_prophets | ancient_prophets | pre_prophetic | early_makkah | late_makkah | madinah_early | madinah_late",
  "region": "one of: makkah | madinah | arabian_desert | egypt_ancient | levant_ancient | sinai | abyssinia | other",
  "source_reference": "scholarly source",
  "characters": [
    {
      "id": "lowercase_id",
      "name": "Full Name",
      "role": "one of: prophet | sahabi | family_of_prophet | antagonist | supporting | crowd",
      "depiction_rule": "one of: NO_FACE | FROM_BEHIND | SILHOUETTE | FULL",
      "appearance": "detailed one-paragraph physical description with clothing colors"
    }
  ],
  "moral_lesson": "the Islamic value or lesson this scene teaches",
  "narrative_text": "simple engaging story text for children aged 6-12, 2-3 sentences",
  "compliance": {
    "prophet_check": "NO_PROPHET_IN_SCENE | PROPHET_FROM_BEHIND | PROPHET_SILHOUETTE | PROPHET_NOT_VISIBLE",
    "modesty_check": "COMPLIANT",
    "child_safe": "APPROPRIATE",
    "notes": "brief explanation of compliance decisions"
  },
  "image_prompt": "80-120 word detailed visual description for an image generation model. Start with art style. Include all character appearances, setting details, lighting, composition. End with 'No modern objects.'"
}"""

# ============================================================
# FEW-SHOT EXAMPLE (included in every call for consistency)
# ============================================================
FEW_SHOT_EXAMPLE = """Here is an example of correct output:

INPUT PASSAGE: "Abu Bakr wept with joy when the Prophet told him he would be his companion on the journey to Madinah. Abu Bakr had already prepared two camels, feeding them for months in anticipation."
SOURCE: Ibn Hisham, Al-Sira, Vol. 2

OUTPUT:
{"scene_title":"Abu Bakr Learns He Will Accompany the Prophet","era":"late_makkah","region":"makkah","source_reference":"Ibn Hisham, Al-Sira Al-Nabawiyya, Vol. 2","characters":[{"id":"prophet_muhammad","name":"Prophet Muhammad (PBUH)","role":"prophet","depiction_rule":"FROM_BEHIND","appearance":"Early 50s, medium build, dignified upright posture. Wearing a simple white thobe with a green cloak draped over shoulders, white turban with tail between shoulder blades. Simple leather sandals. Always shown from behind, face never visible. Distinguished by green cloak and subtle warm light."},{"id":"abu_bakr","name":"Abu Bakr al-Siddiq","role":"sahabi","depiction_rule":"FULL","appearance":"Late 40s, light olive skin, slender build, slightly shorter than average. Thin grey-streaked beard with slight reddish henna tint, neatly kept. Wearing off-white cotton thobe, plain brown woolen cloak, white cloth wrapped as loose head covering. Worn leather sandals. Kind gentle eyes, slightly stooped shoulders."}],"moral_lesson":"True friendship means being ready to sacrifice everything for the sake of Allah and those you love.","narrative_text":"When the Prophet told Abu Bakr that he would be his companion on the journey, Abu Bakr cried tears of joy. He had been preparing for this day — he had already bought two strong camels and fed them well for months!","compliance":{"prophet_check":"PROPHET_FROM_BEHIND","modesty_check":"COMPLIANT","child_safe":"APPROPRIATE","notes":"Prophet shown from behind, backlit by doorway light. Abu Bakr shown with full face, tears visible. Warm emotional scene, appropriate for children."},"image_prompt":"Children's watercolor illustration. Interior of a modest 7th century Makkah mud-brick home. A man in white thobe and green cloak, white turban, seen entirely from behind, standing at an open doorway with warm golden afternoon sunlight streaming behind him. Inside, a slender older Arab man with light olive skin, thin grey-streaked beard with henna tint, off-white thobe and brown cloak, sitting on a woven mat looking up with tears of joy, one hand on his chest. Simple mud-brick walls, clay water jug in corner. Medium shot, eye level. Warm golden palette. No modern objects."}"""


# ============================================================
# API CALLERS
# ============================================================
def call_api(passage_text, source):
    user_prompt = f"""{FEW_SHOT_EXAMPLE}

Now convert this passage:

INPUT PASSAGE: "{passage_text}"
SOURCE: {source}

OUTPUT:"""

    if PROVIDER == "groq":
        from groq import Groq
        client = Groq(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL_ID["groq"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    elif PROVIDER == "together":
        from together import Together
        client = Together(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL_ID["together"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
        )
        return response.choices[0].message.content

    elif PROVIDER == "google":
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(
            MODEL_ID["google"],
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=4000,
                response_mime_type="application/json"
            )
        )
        response = model.generate_content(user_prompt)
        return response.text

    elif PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL_ID["openai"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content


# ============================================================
# VALIDATION
# ============================================================
def validate_output(response_text):
    """Validate the LLM output against v2 schema. Returns (parsed_json, errors)."""
    errors = []

    # Clean markdown fences
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [f"Invalid JSON: {str(e)[:80]}"]

    # Required fields
    for field in ["scene_title", "era", "characters", "narrative_text", "compliance", "image_prompt"]:
        if field not in data:
            errors.append(f"Missing field: {field}")

    # Characters check
    chars = data.get("characters", [])
    if not isinstance(chars, list) or len(chars) == 0:
        errors.append("No characters defined")
    else:
        for c in chars:
            if c.get("role") == "prophet" and c.get("depiction_rule") == "FULL":
                errors.append(f"CRITICAL: Prophet '{c.get('name')}' has FULL depiction rule")
            if not c.get("appearance") or len(c.get("appearance", "")) < 20:
                errors.append(f"Character '{c.get('id')}' has insufficient appearance description")

    # Image prompt check
    prompt = data.get("image_prompt", "")
    if len(prompt.split()) < 40:
        errors.append(f"Image prompt too short ({len(prompt.split())} words, need 60+)")

    return data, errors


# ============================================================
# PROGRESS MANAGEMENT
# ============================================================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ============================================================
# MAIN
# ============================================================
def main():
    if API_KEY == "your-api-key-here":
        print("ERROR: Set your API_KEY at the top of the script!")
        print(f"Current provider: {PROVIDER}")
        print(f"Get a free key:")
        print(f"  Google: https://aistudio.google.com/apikey")
        print(f"  Groq:   https://console.groq.com")
        sys.exit(1)

    # Load passages
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Place it in the same directory.")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        passages = json.load(f)

    print(f"Loaded {len(passages)} passages")

    # Setup output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load progress (for resume)
    progress = load_progress()
    completed_ids = set(progress["completed"])
    remaining = [p for p in passages if p["id"] not in completed_ids]

    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Provider: {PROVIDER} ({MODEL_ID[PROVIDER]})")
    print(f"Delay between calls: {DELAY[PROVIDER]}s")
    print()

    if not remaining:
        print("All passages already processed! Building combined dataset...")
        build_combined_dataset(passages)
        return

    success = 0
    fail = 0
    start_time = time.time()

    for i, passage in enumerate(remaining):
        pid = passage["id"]
        print(f"[{len(completed_ids) + i + 1}/{len(passages)}] {pid}: {passage['chapter_title'][:40]}...", end=" ", flush=True)

        try:
            response = call_api(passage["passage"], passage["source"])
            data, errors = validate_output(response)

            if data is None:
                # Complete failure
                print(f"FAIL (invalid JSON)")
                progress["failed"].append({"id": pid, "errors": errors})
                fail += 1
            elif any("CRITICAL" in e for e in errors):
                # Critical error (prophet face violation)
                print(f"FAIL ({errors[0]})")
                progress["failed"].append({"id": pid, "errors": errors})
                fail += 1
            else:
                # Success (may have minor warnings)
                output = {
                    "training_id": pid,
                    "input": passage["passage"] + f"\nSource: {passage['source']}",
                    "output": data,
                    "metadata": {
                        "chapter": passage["chapter"],
                        "chapter_title": passage["chapter_title"],
                        "scene_types": passage["scene_types"],
                        "generation_warnings": errors if errors else None
                    }
                }

                output_path = os.path.join(OUTPUT_DIR, f"{pid}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)

                progress["completed"].append(pid)
                completed_ids.add(pid)
                success += 1

                status = "OK" if not errors else f"OK (warnings: {len(errors)})"
                prompt_words = len(data.get("image_prompt", "").split())
                print(f"{status}, prompt={prompt_words}w")

        except Exception as e:
            error_msg = str(e)[:100]
            print(f"ERROR: {error_msg}")
            progress["failed"].append({"id": pid, "errors": [error_msg]})
            fail += 1

            # If rate limited, wait longer
            if "rate" in error_msg.lower() or "429" in error_msg:
                print("    Rate limited. Waiting 60s...")
                time.sleep(60)

        # Save progress periodically
        if (i + 1) % 10 == 0:
            save_progress(progress)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            remaining_est = (len(remaining) - i - 1) / rate if rate > 0 else 0
            print(f"    --- Progress saved. {rate:.1f} passages/min, ~{remaining_est:.0f} min remaining ---")

        time.sleep(DELAY[PROVIDER])

    # Final save
    save_progress(progress)

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"COMPLETE")
    print(f"{'='*50}")
    print(f"Success: {success}")
    print(f"Failed:  {fail}")
    print(f"Time:    {elapsed/60:.1f} minutes")
    print(f"Rate:    {(success+fail)/elapsed*60:.1f} passages/min")

    if fail > 0:
        print(f"\nFailed passages saved in {PROGRESS_FILE}")
        print(f"To retry: delete their IDs from 'completed' in {PROGRESS_FILE} and rerun")

    # Build combined dataset
    build_combined_dataset(passages)


def build_combined_dataset(passages):
    """Combine all individual JSONs into one training dataset file."""
    dataset = []
    missing = []

    for passage in passages:
        path = os.path.join(OUTPUT_DIR, f"{passage['id']}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                dataset.append(json.load(f))
        else:
            missing.append(passage["id"])

    output_path = "tariikhna_training_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nCombined dataset: {output_path}")
    print(f"  Total examples: {len(dataset)}")
    if missing:
        print(f"  Missing: {len(missing)} (failed or not yet processed)")

    # Stats
    if dataset:
        avg_prompt = sum(
            len(d["output"].get("image_prompt", "").split())
            for d in dataset
        ) / len(dataset)
        print(f"  Avg image prompt length: {avg_prompt:.0f} words")


if __name__ == "__main__":
    main()
