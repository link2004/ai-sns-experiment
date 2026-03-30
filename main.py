"""
AI-only SNS — Metacognition-based post generation

Architecture:
  Pre-compute: Extract concrete data points (topics, people, thought seeds) from preference_analysis
  Stage -1: Generate writing style profile per user (LLM, once)
  Stage  0: Generate day context per user (LLM, once)
  Stage  1: Generate internal state (LLM, per post) — surgical injection of concrete topics
  Stage  2: Generate post text (LLM, per post) — style profile driven

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
MODEL = "google/gemini-3-flash-preview"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HOURS = [7, 8, 9, 12, 13, 15, 18, 19, 20, 21, 22, 23]
DISPLAY_NAMES = ["wataru", "shuuuu2", "hana", "furukaho", "iii", "cochan16", "riku"]

CHARACTER_URLS = {
    "cochan16": "https://ftrfpbrgnjkqgzaggkdz.supabase.co/storage/v1/object/public/profile-avatar/generated-avatars/b7586fe2-659a-4cef-bb2d-8719ca38a77d/20260214_045427.png",
    "furukaho": "https://ftrfpbrgnjkqgzaggkdz.supabase.co/storage/v1/object/public/profile-avatar/generated-avatars/07c681c2-3310-45bf-8930-dc087eac462c/20260328_035200.png",
    "hana": "https://ftrfpbrgnjkqgzaggkdz.supabase.co/storage/v1/object/public/profile-avatar/generated-avatars/79cc4cdc-d60f-49ea-a738-d0b0ec6abc2c/20260210_102057.png",
    "iii": "https://ftrfpbrgnjkqgzaggkdz.supabase.co/storage/v1/object/public/profile-avatar/generated-avatars/74141642-884d-4169-89ad-d202d1560a19/20260324_185617.png",
    "riku": "https://ftrfpbrgnjkqgzaggkdz.supabase.co/storage/v1/object/public/profile-avatar/generated-avatars/211a7e63-b135-4595-8d28-cb8689cb6617/20260219_063036.png",
    "shuuuu2": "https://ftrfpbrgnjkqgzaggkdz.supabase.co/storage/v1/object/public/profile-avatar/generated-avatars/f9964ff4-7efc-4495-a4e1-af311754dfa8/20260310_214902.png",
    "wataru": "https://ftrfpbrgnjkqgzaggkdz.supabase.co/storage/v1/object/public/profile-avatar/generated-avatars/310d8ad9-e71a-4552-acf5-860332e691d5/20260208_015142.png",
}

ANGLE_HINTS = [
    "五感に注目（見えてるもの、聞こえてる音、匂い）",
    "ふと昔のことを思い出す",
    "身体の感覚（空腹、疲れ、心地よさ）",
    "小さな違和感やイラッとしたこと",
    "小さな幸せや満足感",
    "退屈で何か面白いことないかな",
    "誰かのことをふと考えている",
    "今やってることに集中している",
]


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
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines)
    return json.loads(raw)


# ============================================================
# Pre-compute: Extract concrete data from preference_analysis
# ============================================================


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated value string into items."""
    items = []
    for sep in ["、", "，", ","]:
        if sep in value:
            items = [v.strip() for v in value.split(sep)]
            break
    if not items:
        items = [value.strip()]
    return [i for i in items if i and len(i) < 30]


def _get_list_value(sections: list[dict], section_key: str, label: str) -> str:
    """Get a specific label's value from a list section."""
    for s in sections:
        if s.get("key") == section_key and s.get("type") == "list":
            for item in s.get("body", []):
                if item.get("label") == label:
                    return item.get("value", "")
    return ""


def _get_list_values(sections: list[dict], section_key: str) -> dict[str, str]:
    """Get all label:value pairs from a list section."""
    for s in sections:
        if s.get("key") == section_key and s.get("type") == "list":
            return {item["label"]: item.get("value", "") for item in s.get("body", [])}
    return {}


def extract_topic_pool(sections: list[dict]) -> list[str]:
    """Extract concrete topic strings from interests, foods, places, roots."""
    topics = []

    # interests
    for label in ["推し", "よく見ているコンテンツ", "何回も出てく���キーワード"]:
        val = _get_list_value(sections, "interests", label)
        if val:
            topics.extend(_split_csv(val))

    # food_shopping
    for label in ["好きな食ジャンル", "よく買うもの"]:
        val = _get_list_value(sections, "food_shopping", label)
        if val:
            topics.extend(_split_csv(val))

    # daily_life
    val = _get_list_value(sections, "daily_life", "よくいる場所")
    if val:
        topics.extend(_split_csv(val))

    # roots
    val = _get_list_value(sections, "roots", "思い入れのある場所")
    if val:
        topics.extend(_split_csv(val))

    return list(set(topics))


def extract_people_pool(sections: list[dict]) -> list[str]:
    """Extract friend names from relationships section."""
    val = _get_list_value(sections, "relationships", "よく登場する友達")
    if val:
        return _split_csv(val)
    return []


def extract_thought_seeds(sections: list[dict]) -> list[str]:
    """Extract concrete thought patterns from inner_thoughts."""
    values = _get_list_values(sections, "inner_thoughts")
    return [v for v in values.values() if v and len(v) < 60]


def extract_prose_text(sections: list[dict], key: str) -> str:
    """Extract prose body text, stripping source refs."""
    for s in sections:
        if s.get("key") == key and s.get("type") == "prose":
            return re.sub(r"\{\d+(,\d+)*\}", "", str(s.get("body", "")))
    return ""


# ============================================================
# Layer 1: Load personas
# ============================================================


def load_personas() -> dict[str, dict]:
    """Load preference_analysis JSON files and build enriched persona dicts."""
    files = sorted(DATA_DIR.glob("user_*.json"))
    if not files:
        print(f"Error: No user data files found in {DATA_DIR}")
        sys.exit(1)

    personas = {}
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            sc = json.load(f)

        # Derive name from filename: user_riku.json → riku
        name = fpath.stem.removeprefix("user_")
        if name not in DISPLAY_NAMES:
            continue
        sections = sc.get("sections", [])
        axis = sc.get("axis_scores", {})

        # Pre-compute concrete data pools
        topic_pool = extract_topic_pool(sections)
        people_pool = extract_people_pool(sections)
        thought_seeds = extract_thought_seeds(sections)

        # Derive posting behavior from axis_scores
        def get_score(category: str, axis_id: str) -> int:
            for ax in axis.get(category, []):
                if ax.get("id") == axis_id:
                    return ax.get("score", 3)
            return 3

        sharing = get_score("hobby", "hobby_sharing")
        energy = get_score("friendship", "friend_energy")
        weekend = get_score("hobby", "hobby_weekend")
        depth_score = get_score("work", "work_thinking")

        freq_score = (6 - sharing) + energy
        frequency = "high" if freq_score >= 7 else ("low" if freq_score <= 4 else "medium")

        if weekend <= 2:
            active_hours = [7, 8, 9, 12, 15, 18, 21]
        elif weekend >= 4:
            active_hours = [12, 13, 15, 18, 20, 21, 22, 23]
        else:
            active_hours = [8, 12, 15, 18, 20, 22, 23]

        reaction_rate = min(0.5, max(0.1, (6 - sharing) * 0.1))
        inner_depth = "deep" if depth_score <= 2 else ("shallow" if depth_score >= 4 else "moderate")

        personas[name] = {
            "display_name": name,
            "tagline": sc.get("tagline", ""),
            "tagline_bullets": sc.get("tagline_bullets", []),
            # Concrete data pools (for surgical injection)
            "topic_pool": topic_pool,
            "people_pool": people_pool,
            "thought_seeds": thought_seeds,
            # Full text (for style profile generation only)
            "_personality_prose": extract_prose_text(sections, "personality"),
            "_interests_text": "\n".join(
                f"- {item['label']}: {item.get('value', '')}"
                for s in sections if s.get("key") == "interests"
                for item in s.get("body", [])
            ),
            "_daily_life_text": "\n".join(
                f"- {item['label']}: {item.get('value', '')}"
                for s in sections if s.get("key") == "daily_life"
                for item in s.get("body", [])
            ),
            # Posting behavior
            "behavior": {
                "frequency": frequency,
                "active_hours": active_hours,
                "reaction_rate": reaction_rate,
                "inner_depth": inner_depth,
            },
            # Will be set by Stage -1
            "style_profile": None,
            "_raw": sc,
        }
        print(f"  {name}: {sc.get('tagline', '?')} ({len(topic_pool)} topics, {len(people_pool)} people, {len(thought_seeds)} seeds)")

    return personas


# ============================================================
# Stage -1: Style profile generation (once per user)
# ============================================================

STYLE_SYSTEM = """\
You define a person's SNS writing style based on their real personality data.
Output JSON only. All text in Japanese.
CRITICAL: Exaggerate their personality traits to be FUNNY. Make them a caricature of themselves.
If they like ramen, they should be OBSESSED with ramen. If they study hard, everything becomes a study metaphor.
Think: the funniest version of this person's Twitter. Their friends would laugh reading it."""

STYLE_USER_TEMPLATE = """\
以下の人物のSNS投稿の「文体」を定義してください。

【キャッチコピー】
{tagline}
{tagline_bullets}

【性格・人柄】
{personality}

【関心領域】
{interests}

【日常】
{daily_life}

【この人の具体的なトピック（投稿例に織り込むこと）】
{sample_topics}

以下のJSON形式で出力:
{{
  "first_person": "この人が使う一人称（私/僕/俺/あたし/うち/自分 等）",
  "emoji_style": "絵文字の使い方（例: 使わない / たまに1個 / 文末に1つ程度）",
  "quirks": ["この人特有の表現の癖を3つ（具体的に。例: 理系っぽい比喩を混ぜる）"],
  "tone_keywords": ["文体を表すキーワードを3つ（例: 知的、素朴、エモい）"],
  "never_says": ["この人が絶対言わない表現を5つ"],
  "example_posts": [
    "架空のSNS投稿例1",
    "架空のSNS投稿例2",
    "架空のSNS投稿例3",
    "架空のSNS投稿例4",
    "架空のSNS投稿例5"
  ]
}}

重要:
- example_postsはこの人の具体的なトピック（上記）を必ず含めること
- 投稿例は15〜40文字。1〜2文。句読点少なめ。つぶやき感覚。
- リアルな20代日本人のTwitter/Instagramストーリーの温度感にすること
- 「素敵」「最高」「幸せ」「充実」など陳腐なポジティブワードの多用禁止
- 絵文字は最大1個。使わなくてもいい。2個以上は禁止
- ハッシュタグは使わない
- 5つの投稿例はそれぞれ違うトーン（楽しい/だるい/ぼーっと/ちょっとイラッ/しみじみ）にすること
- 全ての項目をこの人の個性に合わせて具体的に書くこと
- 敬語・丁寧語は絶対禁止（〜です、〜ます、〜ですね、〜してます は使わない）
- タメ口・カジュアルな口調のみ"""


def generate_style_profile(persona: dict) -> dict:
    """Stage -1: Generate writing style profile."""
    topics = persona["topic_pool"]
    sample = ", ".join(random.sample(topics, min(8, len(topics))))

    prompt = STYLE_USER_TEMPLATE.format(
        tagline=persona["tagline"],
        tagline_bullets="\n".join(persona["tagline_bullets"]),
        personality=persona["_personality_prose"][:600],
        interests=persona["_interests_text"][:400],
        daily_life=persona["_daily_life_text"][:300],
        sample_topics=sample,
    )
    return call_llm_json(STYLE_SYSTEM, prompt, temperature=0.8, max_tokens=600)


# ============================================================
# Stage 0: Day context generation
# ============================================================

STAGE0_SYSTEM = """\
You simulate a realistic day for a person. Output JSON only.
The day should be mundane and realistic — not dramatic.
All text in Japanese."""

STAGE0_USER_TEMPLATE = """\
以下の人物の今日一日をシミュレートしてください。

【この人について��
{tagline}

【日常】
{daily_life}

【今日に関わりそうなトピック（これらを出来事に織り込むこと）】
{day_topics}

以下のJSON形式で出力:
{{
  "weather_feeling": "天気に対する感じ方",
  "overall_mood": "今日の気分（1文）",
  "events": [
    {{"hour": 7, "what": "具体的に何が起きたか（1文、上のトピックに関連させて）"}},
    ...
  ],
  "undercurrent": "今日うっすら頭にあること"
}}

eventsは4〜6個。平凡な日常。ドラマチックにしない。
時間は {hours} の中から選ぶ。"""


def generate_day_context(persona: dict) -> dict:
    """Stage 0: Generate day context with concrete topics injected."""
    topics = persona["topic_pool"]
    day_topics = random.sample(topics, min(4, len(topics)))

    prompt = STAGE0_USER_TEMPLATE.format(
        tagline=persona["tagline"],
        daily_life=persona["_daily_life_text"][:300],
        day_topics="、".join(day_topics),
        hours=", ".join(str(h) for h in HOURS),
    )
    return call_llm_json(STAGE0_SYSTEM, prompt, temperature=1.0, max_tokens=500)


# ============================================================
# Posting probability
# ============================================================


def should_post(behavior: dict, hour: int, posts_today: int) -> bool:
    """Decide if user posts at this hour."""
    base = 0.40
    if hour in behavior["active_hours"]:
        base += 0.30
    freq = behavior["frequency"]
    if freq == "high":
        base += 0.15
    elif freq == "low":
        base -= 0.05
    if posts_today >= 6:
        base -= 0.15
    if posts_today >= 6:
        base -= 0.2
    return random.random() < max(0.05, min(base, 0.85))


# ============================================================
# Stage 1: Internal state generation (surgical injection)
# ============================================================

STAGE1_SYSTEM = """\
You simulate the INNER MIND of a person before they post on SNS.
You are NOT generating a post. You generate their internal state.
Output JSON only. All text in Japanese.

Behavioral Guidelines (apply to the character's inner world):
- Their thoughts naturally reflect 3 layers: immediate feelings (L1), patterns they're noticing (L2), and long-term aspirations (L3)
- They value sustainability over short-term wins
- They think about how their actions affect their skills, relationships, and wellbeing
- Rest and learning are seen as strategic investments, not laziness"""

STAGE1_USER_TEMPLATE = """\
この人は今{hour}時。

【今この人の頭にあるトピック（1つだけ。これについてだけ考える）】
- {topic_1}

{person_block}

【この人の思考の癖】
- {thought_seed}
- 思考の深さ: {inner_depth}

【今日の出来事（{hour}時頃）】
{current_event}

{timeline_block}

【今日すでに投稿した内容（繰り返すな）】
{already_posted}

【視点のヒント】
{angle_hint}

JSON出力:
{{
  "wants_to_post": true/false,
  "situation": "今の状況（場所、していること）1文",
  "raw_thought": "投稿前の生の思考。トピック1つだけに絞ること。1文。",
  "emotion": "具体的な感情（1語ではなく状況込みで）",
  "posting_intent": "記録/共感/交流/表現/発散/特になし",
  "reply_to_user": "タイムラインの誰かの投稿に反応する場合、そのユーザー名。反応しない場合はnull"
}}"""


def generate_internal_state(
    persona: dict,
    hour: int,
    day_ctx: dict,
    recent_posts: list[dict],
    already_posted: list[str],
    used_topics: set[str],
) -> tuple[dict, list[str]]:
    """Stage 1: Generate internal state with surgical topic injection.

    Returns (state_dict, selected_topics).
    """
    behavior = persona["behavior"]

    # Select 2 topics (avoid already used ones)
    pool = persona["topic_pool"]
    available = [t for t in pool if t not in used_topics]
    if len(available) < 2:
        available = pool
    selected_topics = random.sample(available, min(2, len(available)))

    # Maybe select a person
    person_block = ""
    people = persona["people_pool"]
    if people and random.random() < 0.4:
        person = random.choice(people)
        person_block = f"【頭の片隅にいる人】\n- {person}"

    # Thought seed
    seeds = persona["thought_seeds"]
    thought_seed = random.choice(seeds) if seeds else "特になし"

    # Current event from day context
    current_event = "特に何もない時間"
    for ev in day_ctx.get("events", []):
        if ev.get("hour") == hour:
            current_event = ev["what"]
            break

    # Timeline: only show with reaction_rate probability (anti-echo-chamber)
    if recent_posts and random.random() < behavior["reaction_rate"]:
        lines = [f"  [{p['time']}] {p['user']}: {p['text']}" for p in recent_posts[-5:]]
        timeline_block = "【タイムライン（最近見た投稿）】\n" + "\n".join(lines)
    else:
        timeline_block = "（タイムラインは見ていない。自分の世界に没頭中）"

    # Already posted
    posted_text = "（まだ投稿していない）"
    if already_posted:
        posted_text = "\n".join(f"  - {t}" for t in already_posted[-3:])

    prompt = STAGE1_USER_TEMPLATE.format(
        hour=hour,
        topic_1=selected_topics[0] if selected_topics else "特になし",
        topic_2=selected_topics[1] if len(selected_topics) > 1 else "特になし",
        person_block=person_block,
        thought_seed=thought_seed,
        inner_depth=behavior["inner_depth"],
        current_event=current_event,
        timeline_block=timeline_block,
        already_posted=posted_text,
        angle_hint=random.choice(ANGLE_HINTS),
    )
    state = call_llm_json(STAGE1_SYSTEM, prompt, temperature=1.0)
    return state, selected_topics


# ============================================================
# Stage 2: Post generation (style profile driven)
# ============================================================

STAGE2_SYSTEM = """\
You write a single FUNNY SNS post as a specific person.
Output ONLY the post text. No explanation, no quotes, no hashtags.
The post MUST be 15-50 characters.
CRITICAL: Be FUNNY. Exaggerate their obsessions. Make their friends laugh.
If they love ramen, EVERYTHING is about ramen. If they code, they see the world as code.
Think: the funniest person in your friend group's Twitter. Absurd, self-aware humor.
EMOJI: Max 1 emoji per post. Zero is fine. No emoji spam."""

STAGE2_USER_TEMPLATE = """\
この人としてSNS投稿を1つ書いてください。

【文体ルール（厳守）】
一人称: {first_person}
絵文字: {emoji_style}
文の長さ: 15〜50文字。1〜2文まで。短いほど良い。
癖: {quirks}
絶対言わない: {never_says}
敬語・丁寧語は絶対禁止。タメ口のみ。

【投稿例（雰囲気の参考。コピーするな。同じ単語を使うな）】
{example_posts}

【今の内面】
状況: {situation}
思考: {raw_thought}
感情: {emotion}
意図: {posting_intent}

ルール:
- 15〜50文字厳守。超えたら失敗。
- 1〜2文まで。ダラダラ書かない。
- ハッシュタグ禁止。
- 「最高」「素敵」「充実」「幸せ」の多用禁止。
- 投稿例の単語やフレーズをそのまま使うな。
- フォロワーに向けた投稿。見た人がクスッと笑える内容。
- この人の「沼」「執着」「あるある」を全面に出す。
- 自虐、大げさ、ツッコミ待ち、なんでもあり。
- 敬語禁止。タメ口のみ。
- 真面目すぎる投稿は失敗。面白さ優先。
- 絵文字は最大1個。なくてもいい。絵文字まみれは禁止。
- 1ツイート1話題。2つ以上の話題を混ぜるな。シンプルに1個だけ。

投稿内容のみ出力。"""


# ============================================================
# Reply generation
# ============================================================

REPLY_SYSTEM = """\
You write a short FUNNY reply to someone's SNS post as a specific person.
Output ONLY the reply text. No explanation, no quotes.
The reply MUST be 10-40 characters. Very short and casual.
NEVER use keigo. Think: roasting your friend, banter, ツッコミ.
Be funny. Tease them. Call them out. Or agree in the most absurd way."""

REPLY_USER_TEMPLATE = """\
{replier_name}として、{poster_name}の投稿に返信してください。

【{replier_name}の文体】
一人称: {first_person}
絵文字: {emoji_style}
敬語禁止。タメ口のみ。

【{poster_name}の投稿】
「{post_text}」

【{replier_name}と{poster_name}の関係】
SNS上の友達。カジュアルな関係。

ルール:
- 10〜40文字。短いほど良い。
- 敬語禁止。タメ口のみ。
- 友達にLINEで返すくらいのノリ。
- ツッコミ、いじり、便乗、からかい、大げさな共感、どれでもOK。
- 面白さ優先。真面目な返信は禁止。
- 絵文字は最大1個。なくていい。
- 「わかる」「それな」だけで終わらない。もう少しふざけて。
- 相手をいじるくらいがちょうどいい。

返信内容のみ出力。"""


def generate_reply(replier: dict, poster_name: str, post_text: str) -> str:
    """Generate a casual reply to someone's post."""
    sp = replier["style_profile"]
    if not sp:
        return "わかるわ〜"

    prompt = REPLY_USER_TEMPLATE.format(
        replier_name=replier["display_name"],
        poster_name=poster_name,
        first_person=sp.get("first_person", "私"),
        emoji_style=sp.get("emoji_style", "控えめ"),
        post_text=post_text,
    )
    text = call_llm(REPLY_SYSTEM, prompt, temperature=0.9, max_tokens=80)
    return text.strip().strip('"').strip("「」")


def generate_replies_for_hour(
    personas: dict,
    all_posts: list[dict],
    hour: int,
) -> list[dict]:
    """Generate replies from other users to recent posts in this hour."""
    replies = []
    # Get posts from this hour
    hour_posts = [
        (i, p) for i, p in enumerate(all_posts)
        if p["time"].startswith(f"{hour:02d}:") and p.get("reply_to_idx") is None
    ]
    if not hour_posts:
        return replies

    for post_idx, post in hour_posts:
        poster_name = post["user"]
        # Each other user has a chance to reply
        for uid, persona in personas.items():
            if persona["display_name"] == poster_name:
                continue  # Don't reply to self
            # Reply probability: ~25% per user per post
            if random.random() > 0.25:
                continue

            try:
                reply_text = generate_reply(persona, poster_name, post["text"])
            except Exception:
                continue

            reply_time = f"{hour:02d}:{random.randint(0, 59):02d}"
            reply = {
                "user": persona["display_name"],
                "time": reply_time,
                "text": reply_text,
                "internal_state": {},
                "reply_to_idx": post_idx,
            }
            replies.append(reply)
            time.sleep(0.3)
            # Max 1-2 replies per post
            if len([r for r in replies if r["reply_to_idx"] == post_idx]) >= 2:
                break

    return replies


def generate_post(persona: dict, internal_state: dict, already_posted: list[str]) -> str:
    """Stage 2: Generate post driven by style profile."""
    sp = persona["style_profile"]
    if not sp:
        return internal_state.get("raw_thought", "...")

    # Build a ban list from previous posts to avoid repetition
    ban_words = set()
    for prev in already_posted[-3:]:
        # Extract katakana/English words that might be repeated catchphrases
        for word in re.findall(r'[ァ-ヶー]{3,}|[A-Za-z]{3,}', prev):
            ban_words.add(word)

    ban_text = ""
    if ban_words:
        ban_text = f"\n- 以下の単語は過去投稿で使用済。使うな: {', '.join(ban_words)}"

    prompt = STAGE2_USER_TEMPLATE.format(
        first_person=sp.get("first_person", "私"),
        emoji_style=sp.get("emoji_style", "控えめ"),
        quirks="、".join(sp.get("quirks", [])),
        never_says="、".join(sp.get("never_says", [])),
        example_posts="\n".join(f"  - {p}" for p in sp.get("example_posts", [])),
        situation=internal_state.get("situation", ""),
        raw_thought=internal_state.get("raw_thought", ""),
        emotion=internal_state.get("emotion", ""),
        posting_intent=internal_state.get("posting_intent", ""),
    ) + ban_text
    text = call_llm(STAGE2_SYSTEM, prompt, temperature=0.9, max_tokens=100)
    return text.strip().strip('"').strip("「」")


# ============================================================
# CLI display
# ============================================================


def print_header(today: str) -> None:
    print(f"\n{'─' * 52}")
    print(f"  AI SNS — {today}")
    print(f"{'─' * 52}\n")


def print_post(post: dict) -> None:
    print(f"  {post['time']}  \033[1m{post['user']}\033[0m")
    print(f"           {post['text']}")
    print()


def print_debug(user_id: str, hour: int, state: dict, topics: list[str]) -> None:
    """Print internal state for debugging."""
    print(
        f"  \033[2m[{hour:02d}h] {user_id} "
        f"topics={topics} "
        f"→ {state.get('posting_intent', '?')} "
        f"(post={state.get('wants_to_post', '?')})\033[0m"
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

    # --- Layer 1: Load personas ---
    print("  \033[2mLoading personas...\033[0m")
    personas = load_personas()
    print()

    # --- Stage -1: Generate style profiles ---
    print("  \033[2mGenerating style profiles (Stage -1)...\033[0m")
    for uid, persona in personas.items():
        try:
            sp = generate_style_profile(persona)
            persona["style_profile"] = sp
            print(f"  \033[2m  {uid}: 一人称={sp.get('first_person', '?')}, "
                  f"語尾={sp.get('sentence_endings', [])[:3]}\033[0m")
            if debug and sp.get("example_posts"):
                for ex in sp["example_posts"]:
                    print(f"  \033[2m    例: {ex}\033[0m")
        except Exception as e:
            print(f"  \033[31m  {uid}: Style profile failed: {e}\033[0m")
            persona["style_profile"] = {
                "first_person": "私",
                "emoji_style": "控えめ",
                "typical_length": "20-60文字",
                "quirks": [],
                "tone_keywords": [],
                "never_says": [],
                "example_posts": [],
            }
        time.sleep(0.3)
    print()

    # --- Stage 0: Generate day contexts ---
    print("  \033[2mGenerating day contexts (Stage 0)...\033[0m")
    day_contexts: dict[str, dict] = {}
    for uid, persona in personas.items():
        try:
            ctx = generate_day_context(persona)
            day_contexts[uid] = ctx
            print(f"  \033[2m  {uid}: {ctx.get('overall_mood', '?')}\033[0m")
        except Exception as e:
            print(f"  \033[31m  {uid}: Stage 0 failed: {e}\033[0m")
            day_contexts[uid] = {
                "weather_feeling": "普通",
                "overall_mood": "普通",
                "events": [],
                "undercurrent": "特になし",
            }
        time.sleep(0.3)

    print(f"\n{'─' * 52}\n")

    # --- Main loop ---
    all_posts: list[dict] = []
    user_post_history: dict[str, list[str]] = {uid: [] for uid in personas}
    used_topics: dict[str, set[str]] = {uid: set() for uid in personas}
    skipped = 0

    for hour in HOURS:
        user_ids = list(personas.keys())
        random.shuffle(user_ids)

        for uid in user_ids:
            persona = personas[uid]
            behavior = persona["behavior"]

            if not should_post(behavior, hour, len(user_post_history[uid])):
                continue

            # Stage 1
            try:
                state, selected_topics = generate_internal_state(
                    persona, hour,
                    day_contexts[uid],
                    all_posts,
                    user_post_history[uid],
                    used_topics[uid],
                )
            except Exception as e:
                if debug:
                    print(f"  \033[31m[{hour:02d}h] {uid} Stage 1 error: {e}\033[0m")
                continue

            if debug:
                print_debug(uid, hour, state, selected_topics)

            if not state.get("wants_to_post", True):
                skipped += 1
                continue

            # Stage 2
            try:
                text = generate_post(persona, state, user_post_history[uid])
            except Exception as e:
                if debug:
                    print(f"  \033[31m[{hour:02d}h] {uid} Stage 2 error: {e}\033[0m")
                continue

            # Find reply target if LLM indicated one
            reply_to_user = state.get("reply_to_user")
            reply_to_idx = None
            if reply_to_user:
                # Find the latest post by that user
                for idx in range(len(all_posts) - 1, -1, -1):
                    if all_posts[idx]["user"] == reply_to_user:
                        reply_to_idx = idx
                        break

            # Track
            used_topics[uid].update(selected_topics)
            time_str = f"{hour:02d}:{random.randint(0, 59):02d}"
            post = {
                "user": persona["display_name"],
                "time": time_str,
                "text": text,
                "internal_state": state,
                "reply_to_idx": reply_to_idx,
            }
            all_posts.append(post)
            user_post_history[uid].append(text)
            print_post(post)
            time.sleep(0.3)

        # --- Reply phase: other users react to this hour's posts ---
        hour_replies = generate_replies_for_hour(personas, all_posts, hour)
        for reply in hour_replies:
            all_posts.append(reply)
            # Find uid for the replier
            for uid_r, p_r in personas.items():
                if p_r["display_name"] == reply["user"]:
                    user_post_history[uid_r].append(reply["text"])
                    break
            print_post(reply)

    # --- Guarantee minimum 2 posts per user ---
    MIN_POSTS = 2
    for uid in personas:
        while len(user_post_history[uid]) < MIN_POSTS:
            persona = personas[uid]
            # Pick a random active hour for this catch-up post
            hour = random.choice(persona["behavior"]["active_hours"])
            try:
                state, selected_topics = generate_internal_state(
                    persona, hour,
                    day_contexts[uid],
                    all_posts,
                    user_post_history[uid],
                    used_topics[uid],
                )
                # Force posting even if wants_to_post is False
                text = generate_post(persona, state, user_post_history[uid])
            except Exception as e:
                if debug:
                    print(f"  \033[31m[catch-up] {uid} error: {e}\033[0m")
                break

            used_topics[uid].update(selected_topics)
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

    # --- Save ---
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
                "style_profile": p["style_profile"],
                "topic_pool": p["topic_pool"][:10],
                "behavior": p["behavior"],
                "avatar_url": CHARACTER_URLS.get(uid),
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
