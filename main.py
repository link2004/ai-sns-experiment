"""
AI-only SNS — Metacognition-based post generation

3-layer context model:
  Layer 1: Persona (static) — who they are + SNS psychology
  Layer 2: Day context (Stage 0) — what kind of day it is (generated once per user)
  Layer 3: Internal state (Stage 1) → Post (Stage 2) — 2-stage generation per post

Usage: uv run python main.py [--debug]
"""

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# ============================================================
# Config
# ============================================================

API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
MODEL = "google/gemini-2.5-flash"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HOURS = [7, 8, 9, 12, 13, 15, 18, 19, 20, 21, 22, 23]

# ============================================================
# Layer 1: Personas (static)
# ============================================================

USERS = {
    "wataru": {
        "display_name": "wataru",
        "identity": "28歳 / 都内IT企業のバックエンドエンジニア",
        "interests": ["コーヒー", "サウナ", "キャンプ", "猫"],
        "tone": "落ち着いたトーン。句読点少なめ。写真のキャプションっぽいことがある。絵文字控えめ。",
        "posting_psychology": {
            "motivation": "self_narration",
            "motivation_desc": "独り言。自分の記録としてのSNS。誰かに見せるためじゃない",
            "audience_awareness": "low",
            "inner_depth": "moderate",
            "emotional_range": "narrow",
            "post_triggers": [
                "一息ついた瞬間（コーヒー、風呂上がり）",
                "いい景色や雰囲気に出会ったとき",
                "物欲が湧いたとき（キャンプギア）",
            ],
        },
        "relationships": {
            "shuuu": "共通の友人経由で知り合い。音楽の趣味が合う",
            "mio": "本の話が合う。落ち着いた雰囲気が似てる",
            "takkun": "正反対のテンションだけど嫌いじゃない",
        },
        "example_posts": [
            "ドリップコーヒーと朝の静けさ。これだけでいい",
            "キャンプ用のランタン40分眺めてた。買ってない",
            "会社近くの自販機、3日連続でBOSSが売り切れてる",
        ],
        "active_hours": [7, 8, 12, 18, 22, 23],
        "post_frequency": "low",
    },
    "shuuu": {
        "display_name": "shuuu",
        "identity": "25歳 / フリーランスのデザイナー / 下北沢在住",
        "interests": ["シティポップ", "ビンテージ古着", "デザイン", "深夜ラジオ"],
        "tone": "ゆるくてカジュアル。ひらがな多め。「〜」多用。感情豊か。テンション高め。",
        "posting_psychology": {
            "motivation": "social_bonding",
            "motivation_desc": "友達と繋がるために呟く。共感を求める。反応があると嬉しい",
            "audience_awareness": "medium",
            "inner_depth": "shallow",
            "emotional_range": "wide",
            "post_triggers": [
                "感情が動いたとき（嬉しい、眠い、お腹すいた）",
                "好きなものに触れたとき（音楽、古着）",
                "誰かの投稿を見て共感したとき",
            ],
        },
        "relationships": {
            "wataru": "年上だけどタメ口。落ち着いてて好き",
            "mio": "感性が近い。カフェとか一緒に行きたい",
            "takkun": "テンション高くて元気もらえる",
        },
        "example_posts": [
            "ねむい〜〜〜まだ14時なのにもう夜の気分",
            "下北で見つけたシャツやばい、、70年代のやつ、、",
            "シティポップ聴きながらの深夜作業、これがぼくの本番",
        ],
        "active_hours": [10, 13, 15, 18, 21, 22, 23],
        "post_frequency": "high",
    },
    "mio": {
        "display_name": "mio",
        "identity": "26歳 / 出版社の編集者",
        "interests": ["読書", "映画", "カフェ巡り", "一人の時間", "散歩"],
        "tone": "丁寧だけど親しみやすい。短文と長文が混在。ふとした気づきを投稿する。本や映画の引用をすることがある。",
        "posting_psychology": {
            "motivation": "self_expression",
            "motivation_desc": "感じたこと・考えたことを言葉にしたい。表現欲。日記のような使い方",
            "audience_awareness": "medium",
            "inner_depth": "deep",
            "emotional_range": "moderate",
            "post_triggers": [
                "本や映画で心が動いたとき",
                "一人の時間にふと考えが浮かんだとき",
                "季節の変化や街の景色に気づいたとき",
                "寂しさを感じたとき（でも直接は言わない）",
            ],
        },
        "relationships": {
            "wataru": "静かな人。本の話ができるのが嬉しい",
            "shuuu": "自分にない軽さがあって元気をもらえる",
            "takkun": "真逆だけどその真っ直ぐさに少し憧れる",
        },
        "example_posts": [
            "帰り道、知らない路地に入ったら古い喫茶店を見つけた。こういう出会いがあるから寄り道はやめられない。",
            "「孤独とは、自分自身との対話の時間である」——最近読んだ本のこの一節がずっと頭にある。",
            "雨の日の編集部、静かで好き。",
        ],
        "active_hours": [8, 12, 18, 20, 21, 22, 23],
        "post_frequency": "medium",
    },
    "takkun": {
        "display_name": "takkun",
        "identity": "30歳 / ラーメン屋の店長",
        "interests": ["ラーメン", "筋トレ", "料理", "早起き"],
        "tone": "元気でストレート。「！」多い。飯テロ的な投稿。筋トレ報告。ポジティブ全開。",
        "posting_psychology": {
            "motivation": "routine_logging",
            "motivation_desc": "毎日の記録。やったこと・食べたものを報告するのが習慣。あまり深く考えてない",
            "audience_awareness": "low",
            "inner_depth": "shallow",
            "emotional_range": "narrow",
            "post_triggers": [
                "筋トレ後の達成感",
                "美味いもの食ったとき",
                "仕事がうまくいったとき",
                "朝起きた瞬間のやる気",
            ],
        },
        "relationships": {
            "wataru": "サウナ仲間。もっと一緒に行きたい",
            "shuuu": "若いのに頑張ってる。飯食わせたい",
            "mio": "お客さんで来てほしい",
        },
        "example_posts": [
            "5時起き！ベンチプレス100kg×5達成！最高の朝！",
            "今日のチャーシュー過去イチの出来。写真撮るの忘れた",
            "閉店後の掃除完了。明日も5時起き。おやすみ！",
        ],
        "active_hours": [5, 6, 7, 12, 15, 21, 22],
        "post_frequency": "medium",
    },
}

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
{name}（{identity}）の今日一日をシミュレートしてください。
趣味・興味: {interests}

以下のJSON形式で出力:
{{
  "weather_feeling": "天気に対する個人的な感じ方（例: 曇りだけど嫌いじゃない）",
  "overall_mood": "今日の基調となる気分（1文）",
  "events": [
    {{"hour": 7, "what": "具体的に何が起きたか（1文）"}},
    ...
  ],
  "undercurrent": "今日一日うっすら頭にあること（悩み、楽しみ、気がかり）"
}}

eventsは4〜6個。平凡な日常でOK。ドラマチックにしすぎない。
時間は {hours} の中から選ぶこと。"""


def generate_day_context(user_id: str, user: dict) -> dict:
    """Stage 0: Generate day context for a user."""
    prompt = STAGE0_USER_TEMPLATE.format(
        name=user["display_name"],
        identity=user["identity"],
        interests="、".join(user["interests"]),
        hours=", ".join(str(h) for h in HOURS),
    )
    return call_llm_json(STAGE0_SYSTEM, prompt, temperature=1.2)


# ============================================================
# Posting probability
# ============================================================


def should_post(user: dict, hour: int, posts_today: int) -> bool:
    """Decide if user posts at this hour based on persona."""
    base = 0.25
    if hour in user["active_hours"]:
        base += 0.3
    freq = user["post_frequency"]
    if freq == "high":
        base += 0.15
    elif freq == "low":
        base -= 0.1
    # Fatigue: reduce probability as posts accumulate
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
{name}は今{hour}時。

【この人のSNS心理】
投稿動機: {motivation_desc}
他人の目をどれくらい意識するか: {audience_awareness}
思考の深さ: {inner_depth}
感情の振れ幅: {emotional_range}
投稿したくなる瞬間: {post_triggers}

【今日の状況】
天気の印象: {weather_feeling}
今日の気分: {overall_mood}
頭の片隅にあること: {undercurrent}

【この時間に起きていること】
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
    user: dict,
    hour: int,
    day_ctx: dict,
    recent_posts: list[dict],
    already_posted: list[str],
) -> dict:
    """Stage 1: Generate internal state."""
    psych = user["posting_psychology"]

    # Find event for this hour
    current_event = "特になし"
    for ev in day_ctx.get("events", []):
        if ev.get("hour") == hour:
            current_event = ev["what"]
            break

    # Build timeline text
    timeline = "（まだ誰も投稿していない）"
    if recent_posts:
        lines = [f"  [{p['time']}] {p['user']}: {p['text']}" for p in recent_posts[-8:]]
        timeline = "\n".join(lines)

    # Already posted
    posted_text = "（まだ投稿していない）"
    if already_posted:
        posted_text = "\n".join(f"  - {t}" for t in already_posted)

    prompt = STAGE1_USER_TEMPLATE.format(
        name=user["display_name"],
        hour=hour,
        motivation_desc=psych["motivation_desc"],
        audience_awareness=psych["audience_awareness"],
        inner_depth=psych["inner_depth"],
        emotional_range=psych["emotional_range"],
        post_triggers="、".join(psych["post_triggers"]),
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
{name}としてSNS投稿を1つ書いてください。

【文体】
{tone}

【参考: この人の過去の投稿例（コピーするな。雰囲気だけ参考に）】
{example_posts}

【今の内面状態】
状況: {situation}
生の思考: {raw_thought}
感情: {emotion}
投稿の意図: {posting_intent}

ルール:
- 内面の全てを書く必要はない。SNSに載せる部分だけ自然に
- 投稿の意図に合った書き方にする
  - 記録 → 淡々と事実。飾らない
  - 共感 → 「わかる」を引き出す感じ
  - 交流 → @メンションや問いかけ（タイムラインに相手の投稿がある場合のみ）
  - 表現 → 詩的、文学的、比喩
  - 発散 → 感情をそのまま出す
- 最大100文字程度
- ハッシュタグは使わない
- 投稿内容のみ出力"""


def generate_post(
    user_id: str,
    user: dict,
    internal_state: dict,
) -> str:
    """Stage 2: Generate post from internal state."""
    prompt = STAGE2_USER_TEMPLATE.format(
        name=user["display_name"],
        tone=user["tone"],
        example_posts="\n".join(f"  - {p}" for p in user["example_posts"]),
        situation=internal_state.get("situation", ""),
        raw_thought=internal_state.get("raw_thought", ""),
        emotion=internal_state.get("emotion", ""),
        posting_intent=internal_state.get("posting_intent", ""),
    )
    text = call_llm(STAGE2_SYSTEM, prompt, temperature=0.9, max_tokens=200)
    # Clean up: remove surrounding quotes if present
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
    print(f"  \033[2m[{hour:02d}h] {user_id} inner: {state.get('raw_thought', '?')[:60]}"
          f" → {state.get('posting_intent', '?')}"
          f" (post={state.get('wants_to_post', '?')})\033[0m")


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

    # --- Stage 0: Generate day contexts ---
    print("  \033[2mGenerating day contexts...\033[0m")
    day_contexts: dict[str, dict] = {}
    for uid, user in USERS.items():
        try:
            ctx = generate_day_context(uid, user)
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
    user_post_history: dict[str, list[str]] = {uid: [] for uid in USERS}
    skipped = 0

    for hour in HOURS:
        user_ids = list(USERS.keys())
        random.shuffle(user_ids)

        for uid in user_ids:
            user = USERS[uid]

            # Probability check
            if not should_post(user, hour, len(user_post_history[uid])):
                continue

            # Stage 1: Internal state
            try:
                state = generate_internal_state(
                    uid, user, hour,
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
                text = generate_post(uid, user, state)
            except Exception as e:
                if debug:
                    print(f"  \033[31m[{hour:02d}h] {uid} Stage 2 error: {e}\033[0m")
                continue

            time_str = f"{hour:02d}:{random.randint(0, 59):02d}"
            post = {
                "user": user["display_name"],
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
    for uid in USERS:
        count = len(user_post_history[uid])
        print(f"    {uid}: {count} posts")
    print(f"{'─' * 52}")

    # --- Save results to JSON ---
    output_dir = BASE_DIR / "pages"
    output_dir.mkdir(exist_ok=True)
    output = {
        "date": today,
        "model": MODEL,
        "users": {uid: {
            "display_name": u["display_name"],
            "identity": u["identity"],
            "interests": u["interests"],
            "tone": u["tone"],
            "posting_psychology": u["posting_psychology"],
            "example_posts": u["example_posts"],
        } for uid, u in USERS.items()},
        "day_contexts": day_contexts,
        "posts": all_posts,
    }
    data_path = output_dir / "data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Data saved to {data_path}")


if __name__ == "__main__":
    main()
