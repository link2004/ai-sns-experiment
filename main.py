"""
AI-only SNS — Metacognition-based post generation

3-layer context model:
  Layer 1: Persona from real preference_analysis data (structured_contents)
  Layer 2: Day context (Stage 0) — generated from persona's daily_life/interests
  Layer 3: Internal state (Stage 1) → Post (Stage 2)

Usage: uv run python main.py [--debug]
"""

import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ============================================================
# Config
# ============================================================

API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
MODEL = "google/gemini-2.5-flash"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HOURS = [7, 8, 9, 12, 13, 15, 18, 19, 20, 21, 22, 23]

# Anonymous display names for real users
DISPLAY_NAMES = ["wataru", "shuuu", "mio", "takkun"]

# Stage 1 angle hints — forces different perspective each time
ANGLE_HINTS = [
    "五感に注目して（今見えてるもの、聞こえてる音、匂い）",
    "さっき見たタイムラインの投稿から連想して",
    "ふと昔のことを思い出す瞬間",
    "身体の感覚に注目して（空腹、疲れ、心地よさ）",
    "小さな違和感やイラッとしたこと",
    "小さな幸せや満足感",
    "退屈で何か面白いことないかなと思ってる",
    "誰かのことをふと考えている",
]


# ============================================================
# Layer 1: Load personas from preference_analysis data
# ============================================================


def extract_section(sections: list[dict], key: str) -> dict | None:
    """Extract a section by key from structured_contents."""
    for s in sections:
        if s.get("key") == key:
            return s
    return None


def section_to_text(section: dict | None) -> str:
    """Convert a section to readable text."""
    if not section:
        return ""
    body = section.get("body", "")
    if section.get("type") == "prose":
        # Strip source refs like {1,5,9}
        return re.sub(r"\{\d+(,\d+)*\}", "", str(body))
    if section.get("type") == "list" and isinstance(body, list):
        lines = []
        for item in body:
            label = item.get("label", "")
            value = item.get("value", "")
            lines.append(f"- {label}: {value}")
        return "\n".join(lines)
    return str(body)


def axis_scores_summary(axis_scores: dict) -> str:
    """Summarize axis_scores into readable text for prompts."""
    lines = []
    for category, axes in axis_scores.items():
        if not isinstance(axes, list):
            continue
        for ax in axes:
            score = ax.get("score", 3)
            if score <= 2:
                tendency = ax.get("low_label", "")
            elif score >= 4:
                tendency = ax.get("high_label", "")
            else:
                continue  # Skip neutral scores
            axis_name = ax.get("axis", ax.get("id", ""))
            desc = ax.get("description", "")
            lines.append(f"- {axis_name}: {tendency}（{desc}）")
    return "\n".join(lines) if lines else "特になし"


def load_personas() -> dict[str, dict]:
    """Load preference_analysis JSON files and build persona dicts."""
    files = sorted(DATA_DIR.glob("user_*.json"))
    if not files:
        print(f"Error: No user data files found in {DATA_DIR}")
        sys.exit(1)

    personas = {}
    for i, fpath in enumerate(files[: len(DISPLAY_NAMES)]):
        with open(fpath, encoding="utf-8") as f:
            sc = json.load(f)

        name = DISPLAY_NAMES[i]
        sections = sc.get("sections", [])
        axis = sc.get("axis_scores", {})

        # Extract all sections
        personality = extract_section(sections, "personality")
        roots = extract_section(sections, "roots")
        interests = extract_section(sections, "interests")
        daily_life = extract_section(sections, "daily_life")
        food_shopping = extract_section(sections, "food_shopping")
        relationships = extract_section(sections, "relationships")
        romance = extract_section(sections, "romance")
        inner_thoughts = extract_section(sections, "inner_thoughts")
        future = extract_section(sections, "future")

        personas[name] = {
            "display_name": name,
            "tagline": sc.get("tagline", ""),
            "tagline_bullets": sc.get("tagline_bullets", []),
            # Full sections as text (for prompts)
            "personality": section_to_text(personality),
            "roots": section_to_text(roots),
            "interests": section_to_text(interests),
            "daily_life": section_to_text(daily_life),
            "food_shopping": section_to_text(food_shopping),
            "relationships": section_to_text(relationships),
            "romance": section_to_text(romance),
            "inner_thoughts": section_to_text(inner_thoughts),
            "future": section_to_text(future),
            # Axis scores summary (non-neutral tendencies only)
            "axis_summary": axis_scores_summary(axis),
            # Raw structured data (for JSON output)
            "_raw": sc,
        }
        print(f"  Loaded {name}: {sc.get('tagline', '?')}")

    return personas


# ============================================================
# LLM call wrapper
# ============================================================


def call_llm(
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 300,
) -> str:
    """Thin wrapper around OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"exclude": True},
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        try:
            resp = requests.post(API_URL, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            raise


def call_llm_json(system: str, user: str, **kwargs) -> dict:
    """Call LLM and parse JSON response."""
    raw = call_llm(system, user, json_mode=True, **kwargs)
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines)
    return json.loads(raw)


# ============================================================
# Stage 0: Day context generation (once per user per day)
# ============================================================

STAGE0_SYSTEM = """\
You simulate a realistic day for a person. Output JSON only.
The day should be mundane and realistic — not dramatic.
All text in Japanese."""

STAGE0_USER_TEMPLATE = """\
以下の人物の今日一日をシミュレートしてください。

【この人物について】
{tagline}
{tagline_bullets}

【日常の過ごし方】
{daily_life}

【関心領域】
{interests}

【食・買い物の癖】
{food_shopping}

【性格の特徴的な傾向】
{axis_summary_excerpt}

以下のJSON形式で出力:
{{
  "weather_feeling": "天気に対する個人的な感じ方",
  "overall_mood": "今日の基調となる気分（1文）",
  "events": [
    {{"hour": 7, "what": "具体的に何が起きたか（1文）"}},
    ...
  ],
  "undercurrent": "今日一日うっすら頭にあること（悩み、楽しみ、気がかり）"
}}

eventsは4〜6個。平凡な日常でOK。ドラマチックにしすぎない。
この人の性格・生活リズム・好みに合った出来事にすること。
時間は {hours} の中から選ぶこと。"""


def generate_day_context(user_id: str, persona: dict) -> dict:
    """Stage 0: Generate day context from preference_analysis data."""
    # Pick a random subset of axis scores for variety
    axis_lines = persona["axis_summary"].split("\n")
    excerpt = "\n".join(random.sample(axis_lines, min(8, len(axis_lines))))

    prompt = STAGE0_USER_TEMPLATE.format(
        tagline=persona["tagline"],
        tagline_bullets="\n".join(persona["tagline_bullets"]),
        daily_life=persona["daily_life"],
        interests=persona["interests"],
        food_shopping=persona["food_shopping"],
        axis_summary_excerpt=excerpt,
        hours=", ".join(str(h) for h in HOURS),
    )
    return call_llm_json(STAGE0_SYSTEM, prompt, temperature=1.2)


# ============================================================
# Posting probability (derived from axis_scores)
# ============================================================


def derive_posting_behavior(persona: dict) -> dict:
    """Derive SNS posting behavior from axis_scores."""
    raw = persona.get("_raw", {})
    axis = raw.get("axis_scores", {})

    # Extract relevant axes
    def get_score(category: str, axis_id: str) -> int:
        for ax in axis.get(category, []):
            if ax.get("id") == axis_id:
                return ax.get("score", 3)
        return 3

    # hobby_sharing: 1=すぐに共有, 5=一人で静かに楽しむ
    sharing = get_score("hobby", "hobby_sharing")
    # friend_energy: 1=まったり, 5=ハイテンション
    energy = get_score("friendship", "friend_energy")
    # hobby_spontaneity: 1=その場で決める, 5=先に決めておく
    spontaneity = get_score("hobby", "hobby_spontaneity")
    # hobby_weekend: 1=朝から予定詰める, 5=布団から出ない
    weekend = get_score("hobby", "hobby_weekend")

    # Posting frequency: sharers + energetic people post more
    freq_score = (6 - sharing) + energy  # 2-10 range
    if freq_score >= 7:
        frequency = "high"
    elif freq_score <= 4:
        frequency = "low"
    else:
        frequency = "medium"

    # Active hours: morning types vs night types
    if weekend <= 2:  # Active, schedule-packed
        active_hours = [7, 8, 9, 12, 15, 18, 21]
    elif weekend >= 4:  # Lazy, late riser
        active_hours = [12, 13, 15, 18, 20, 21, 22, 23]
    else:
        active_hours = [8, 12, 15, 18, 20, 22, 23]

    return {
        "frequency": frequency,
        "active_hours": active_hours,
        "sharing_tendency": sharing,
        "energy_level": energy,
        "spontaneity": spontaneity,
    }


def should_post(persona: dict, behavior: dict, hour: int, posts_today: int) -> bool:
    """Decide if user posts at this hour."""
    base = 0.25
    if hour in behavior["active_hours"]:
        base += 0.3
    freq = behavior["frequency"]
    if freq == "high":
        base += 0.15
    elif freq == "low":
        base -= 0.1
    # Fatigue
    if posts_today >= 4:
        base -= 0.15
    if posts_today >= 6:
        base -= 0.2
    return random.random() < max(0.05, min(base, 0.85))


# ============================================================
# Stage 1: Internal state generation
# ============================================================

STAGE1_SYSTEM = """\
You simulate the INNER MIND of a person before they decide whether to post on SNS.
You are NOT generating a post. You are generating their internal state.
Output JSON only. All text in Japanese."""

STAGE1_USER_TEMPLATE = """\
この人は今{hour}時。

【性格・人柄】
{personality}

【内面・思考パターン】
{inner_thoughts}

【人間関係の特徴】
{relationships}

【恋愛との距離】
{romance}

【性格傾向スコア（抜粋���】
{axis_excerpt}

【今日の状況】
天気の印象: {weather_feeling}
今日の気分: {overall_mood}
頭の片隅にあること: {undercurrent}

【この時間に起きていること���
{current_event}

【タイムライン（最近見た投稿）】
{timeline}

【今日すでに投稿した内容（繰り返すな）】
{already_posted}

【今回の視点】
{angle_hint}

この人の「今の内面」をJSON出力してください:
{{
  "wants_to_post": true/false,
  "situation": "今の物理的な状況（場所、していること）",
  "raw_thought": "投稿になる前の生の思考。内面の声。1〜2文",
  "emotion": "具体的な感情（「嬉しい」ではなく「小さな達成感」のように具体的に）",
  "posting_intent": "記録/共感/交流/表現/発散/特になし のいずれか"
}}"""


def generate_internal_state(
    user_id: str,
    persona: dict,
    hour: int,
    day_ctx: dict,
    recent_posts: list[dict],
    already_posted: list[str],
) -> dict:
    """Stage 1: Generate internal state from preference_analysis persona."""
    # Find event for this hour
    current_event = "特になし"
    for ev in day_ctx.get("events", []):
        if ev.get("hour") == hour:
            current_event = ev["what"]
            break

    # Build timeline text
    timeline = "（まだ誰も投稿し���いない）"
    if recent_posts:
        lines = [f"  [{p['time']}] {p['user']}: {p['text']}" for p in recent_posts[-8:]]
        timeline = "\n".join(lines)

    # Already posted
    posted_text = "（まだ投稿していない）"
    if already_posted:
        posted_text = "\n".join(f"  - {t}" for t in already_posted)

    # Random axis excerpt for variety
    axis_lines = persona["axis_summary"].split("\n")
    axis_excerpt = "\n".join(random.sample(axis_lines, min(6, len(axis_lines))))

    prompt = STAGE1_USER_TEMPLATE.format(
        hour=hour,
        personality=persona["personality"][:800],
        inner_thoughts=persona["inner_thoughts"][:600],
        relationships=persona["relationships"][:400],
        romance=persona["romance"][:300],
        axis_excerpt=axis_excerpt,
        weather_feeling=day_ctx.get("weather_feeling", "普通"),
        overall_mood=day_ctx.get("overall_mood", "普通"),
        undercurrent=day_ctx.get("undercurrent", "特になし"),
        current_event=current_event,
        timeline=timeline,
        already_posted=posted_text,
        angle_hint=random.choice(ANGLE_HINTS),
    )
    return call_llm_json(STAGE1_SYSTEM, prompt, temperature=1.0)


# ============================================================
# Stage 2: Post generation from internal state
# ============================================================

STAGE2_SYSTEM = """\
You write a single SNS post as a specific person.
Output ONLY the post text. No explanation, no quotes around it."""

STAGE2_USER_TEMPLATE = """\
この人としてSNS投稿を1つ書いてください。

【この人のキャッチコピー】
{tagline}
{tagline_bullets}

【性格・人柄（文体の参考に）】
{personality_excerpt}

【今の内面状態】
状況: {situation}
生の思考: {raw_thought}
感情: {emotion}
投稿の意図: {posting_intent}

ルール:
- この人の性格・人柄に合った自然な文体で書く
- 内面の全てを書く必要はない。SNSに載せる部分だけ
- 投稿の意図に合った書き方にする
  - 記録 → 淡々と事実。飾らない
  - 共感 → 「わかる」を引き出す感じ
  - 交流 → @メンションや問いかけ（タイムラインに相手の投稿がある場合のみ）
  - 表現 → 詩的、文学的、比喩
  - 発�� → 感情をそのまま出す
- 最大100文字程度
- ハッシュタグは使わない
- 投稿内容のみ出力"""


def generate_post(
    user_id: str,
    persona: dict,
    internal_state: dict,
) -> str:
    """Stage 2: Generate post from internal state + persona."""
    prompt = STAGE2_USER_TEMPLATE.format(
        tagline=persona["tagline"],
        tagline_bullets="\n".join(persona["tagline_bullets"][:2]),
        personality_excerpt=persona["personality"][:400],
        situation=internal_state.get("situation", ""),
        raw_thought=internal_state.get("raw_thought", ""),
        emotion=internal_state.get("emotion", ""),
        posting_intent=internal_state.get("posting_intent", ""),
    )
    text = call_llm(STAGE2_SYSTEM, prompt, temperature=0.9, max_tokens=200)
    return text.strip().strip('"').strip("「」")


# ============================================================
# CLI display
# ============================================================


def print_header(today: str) -> None:
    print(f"\n{'─' * 52}")
    print(f"  AI SNS — {today}")
    print(f"{'─' * 52}\n")


def print_post(post: dict) -> None:
    name = post["user"]
    time_str = post["time"]
    text = post["text"]
    print(f"  {time_str}  \033[1m{name}\033[0m")
    print(f"           {text}")
    print()


def print_stage1_debug(user_id: str, hour: int, state: dict) -> None:
    """Print internal state for debugging (dimmed)."""
    print(
        f"  \033[2m[{hour:02d}h] {user_id} inner: "
        f"{state.get('raw_thought', '?')[:60]}"
        f" → {state.get('posting_intent', '?')}"
        f" (post={state.get('wants_to_post', '?')})\033[0m"
    )


# ============================================================
# Main
# ============================================================


def main():
    if not API_KEY:
        print("Error: OPEN_ROUTER_API_KEY is not set")
        sys.exit(1)

    debug = "--debug" in sys.argv
    today = datetime.now().strftime("%Y-%m-%d")
    print_header(today)

    # --- Layer 1: Load personas from preference_analysis ---
    print("  \033[2mLoading personas from preference_analysis data...\033[0m")
    personas = load_personas()
    print()

    # Derive posting behaviors from axis_scores
    behaviors = {uid: derive_posting_behavior(p) for uid, p in personas.items()}
    if debug:
        for uid, b in behaviors.items():
            print(f"  \033[2m  {uid}: freq={b['frequency']}, "
                  f"active={b['active_hours']}\033[0m")
        print()

    # --- Stage 0: Generate day contexts ---
    print("  \033[2mGenerating day contexts...\033[0m")
    day_contexts: dict[str, dict] = {}
    for uid, persona in personas.items():
        try:
            ctx = generate_day_context(uid, persona)
            day_contexts[uid] = ctx
            print(f"  \033[2m  {uid}: {ctx.get('overall_mood', '?')}\033[0m")
        except Exception as e:
            print(f"  \033[31m  {uid}: Stage 0 failed: {e}\033[0m")
            day_contexts[uid] = {
                "weather_feeling": "普通の天気",
                "overall_mood": "普通",
                "events": [],
                "undercurrent": "特になし",
            }
        time.sleep(0.3)

    print(f"\n{'─' * 52}\n")

    # --- Main loop: simulate the day ---
    all_posts: list[dict] = []
    user_post_history: dict[str, list[str]] = {uid: [] for uid in personas}
    skipped = 0

    for hour in HOURS:
        user_ids = list(personas.keys())
        random.shuffle(user_ids)

        for uid in user_ids:
            persona = personas[uid]
            behavior = behaviors[uid]

            # Probability check
            if not should_post(persona, behavior, hour, len(user_post_history[uid])):
                continue

            # Stage 1: Internal state
            try:
                state = generate_internal_state(
                    uid,
                    persona,
                    hour,
                    day_contexts[uid],
                    all_posts,
                    user_post_history[uid],
                )
            except Exception as e:
                if debug:
                    print(f"  \033[31m[{hour:02d}h] {uid} Stage 1 error: {e}\033[0m")
                continue

            if debug:
                print_stage1_debug(uid, hour, state)

            # Check wants_to_post
            if not state.get("wants_to_post", True):
                skipped += 1
                continue

            # Stage 2: Generate post
            try:
                text = generate_post(uid, persona, state)
            except Exception as e:
                if debug:
                    print(f"  \033[31m[{hour:02d}h] {uid} Stage 2 error: {e}\033[0m")
                continue

            time_str = f"{hour:02d}:{random.randint(0, 59):02d}"
            post = {
                "user": persona["display_name"],
                "time": time_str,
                "text": text,
                "internal_state": state,
            }
            all_posts.append(post)
            user_post_history[uid].append(text)
            print_post(post)

            time.sleep(0.3)

    # --- Summary ---
    print(f"{'─' * 52}")
    print(f"  Total posts: {len(all_posts)}  (skipped: {skipped})")
    for uid in personas:
        count = len(user_post_history[uid])
        tagline = personas[uid]["tagline"]
        print(f"    {uid} ({tagline}): {count} posts")
    print(f"{'─' * 52}")

    # --- Save results to JSON ---
    output_dir = BASE_DIR / "docs"
    output_dir.mkdir(exist_ok=True)
    output = {
        "date": today,
        "model": MODEL,
        "users": {
            uid: {
                "display_name": p["display_name"],
                "tagline": p["tagline"],
                "tagline_bullets": p["tagline_bullets"],
                "personality": p["personality"][:500],
                "interests": p["interests"],
                "daily_life": p["daily_life"],
                "inner_thoughts": p["inner_thoughts"],
            }
            for uid, p in personas.items()
        },
        "day_contexts": day_contexts,
        "posts": all_posts,
    }
    data_path = output_dir / "data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Data saved to {data_path}")


if __name__ == "__main__":
    main()
