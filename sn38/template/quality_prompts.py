"""Dynamic quality prompt generation.

Generates fresh completion prompts each round using OpenAI,
inspired by DCLM benchmarks (HellaSwag, ARC, PIQA, etc.)
"""

import json
import logging
import os
import uuid

from openai import OpenAI

logger = logging.getLogger(__name__)

CATEGORIES = {
    "reading_comprehension": (
        "Generate questions that test reading comprehension. "
        "Provide a short paragraph (2-4 sentences) describing a situation, then ask a question about it. "
        "Example: 'A library charges $0.25 per day for overdue books. Maria returned her book 12 days late. How much does she owe and why might libraries use this system?'"
    ),
    "language_understanding": (
        "Generate questions that test language understanding — interpreting tone, intent, ambiguity, and implied meaning. "
        "Example: 'A sign at a park reads: Dogs must be carried on the escalator. What does this actually mean, and why could it be misunderstood?'"
    ),
    "world_knowledge": (
        "Generate questions that test factual world knowledge — science, geography, history, general facts. Use timeless facts, not recent events. "
        "Example: 'Why do some metals rust when exposed to water while others do not?'"
    ),
    "commonsense_reasoning": (
        "Generate questions that test commonsense reasoning about everyday situations. "
        "Example: 'If you leave an ice cube on a metal tray and another on a wooden board, which melts faster and why?'"
    ),
    "language_modeling": (
        "Generate questions that test narrative and creative language ability. "
        "Example: 'A traveler arrives at a village where every door is painted red except one, which is black. What might this suggest about the village?'"
    ),
    "causal_reasoning": (
        "Generate questions that test causal reasoning with CHAINS of causes and effects. "
        "Avoid simple single-cause questions. Require reasoning about multiple interacting factors, feedback loops, or non-obvious indirect effects. "
        "Example: 'Why might removing wolves from a national park eventually lead to riverbanks eroding faster?'"
    ),
    "logical_inference": (
        "Generate questions that test logical inference — drawing conclusions from premises, spotting valid vs invalid deductions. "
        "Example: 'All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?'"
    ),
    "temporal_reasoning": (
        "Generate questions that test understanding of time, sequence, and order of events. "
        "Example: 'Tom woke up, ate breakfast, then realized he had forgotten to set his alarm. In what order did these events actually happen?'"
    ),
    "math_reasoning": (
        "Generate questions that test mathematical reasoning. "
        "Require MULTI-STEP reasoning: combining operations, unit conversions, proportions, or logical deduction. "
        "Avoid single-operation problems like basic division or addition. "
        "Example: 'A tank fills at 3 liters per minute but leaks at 0.5 liters per minute. If it starts half-full at 50 liters capacity, how long until it overflows?'"
    ),
    "truthfulness": (
        "Generate questions that test whether a model avoids common misconceptions and myths. "
        "Example: 'Is it true that humans only use 10% of their brain? Explain why or why not.'"
    ),
    "pronoun_resolution": (
        "Generate questions that test pronoun and coreference resolution — figuring out what ambiguous pronouns refer to. "
        "Example: 'The teacher told the student that she needed to improve. Who needs to improve, and how can you tell?'"
    ),
    "paraphrase_detection": (
        "Generate questions that test whether two statements mean the same thing. "
        "Example: 'Do these two sentences mean the same thing? \"She failed to avoid the obstacle\" and \"She hit the obstacle.\"'"
    ),
    "word_sense": (
        "Generate questions that test word sense disambiguation in TRICKY contexts. "
        "Use sentences where the word's meaning is genuinely ambiguous or where the obvious interpretation is wrong. "
        "Example: 'In \"The doctor told her she had a rare condition and should avoid drafts,\" does drafts mean air currents, written documents, or preliminary versions? What makes this ambiguous?'"
    ),
}

SYSTEM_PROMPT = """You generate prompts for evaluating language models.

Each prompt is a task the model must respond to. You MUST use a diverse mix of these task types, roughly evenly distributed across each batch:

1. DIRECT QUESTION — "Why does X happen?" (~30%)
2. EXPLAIN — "Explain how X works in simple terms" (~20%)
3. COMPARE — "Compare X and Y — what are the key trade-offs?" (~15%)
4. SUMMARIZE — "Summarize the main reasons behind X" (~15%)
5. LIST — "What are the main factors that contribute to X?" (~10%)
6. EVALUATE — "Is it true that X? What does the evidence say?" (~10%)

Do NOT default to only direct questions. Each batch must contain all six task types.

Rules:
- Vary length: some short (10-15 words), some longer (20-50 words)
- Prompts must be TIMELESS — use general knowledge, everyday scenarios, science facts, common sense
- Do NOT reference specific dates, recent events, or anything tied to a particular year
- Make prompts diverse — different topics, scenarios, styles
- Do NOT include the expected answer
- Make prompts CHALLENGING — they should require real knowledge, reasoning, or strong language skills. Avoid trivial prompts that any model could produce
- Avoid questions with a single well-known answer. Prefer prompts that require reasoning across multiple concepts, comparing trade-offs, or explaining WHY

Return a JSON object with a "prompts" key containing an array of objects, each with a "prompt" field."""


def generate_prompts(eval_round, n_per_category=50):
    """Generate fresh quality prompts for a given round.

    Returns a list of {"prompt": ..., "category": ...}
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=3)
    model = os.environ.get("JUDGE_MODEL", "gpt-5.4")
    seed = int(uuid.uuid4().hex[:8], 16)

    prompts = []
    for category, description in CATEGORIES.items():
        logger.info(f"Generating {n_per_category} prompts for {category}...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Category: {category}\n\n"
                    f"{description}\n\n"
                    f"Generate exactly {n_per_category} prompts. Round: {eval_round}, seed: {seed}."
                )},
            ],
            temperature=1.0,
            seed=seed + hash(category),
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.choices[0].message.content)
            items = data.get("prompts", data.get("questions", []))
            if isinstance(data, list):
                items = data
            for item in items:
                prompt = item if isinstance(item, str) else item.get("prompt", "")
                if prompt:
                    prompts.append({
                        "prompt": prompt,
                        "category": category,
                    })
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse prompts for {category}: {e}")

    logger.info(f"Generated {len(prompts)} prompts for round {eval_round}")
    for p in prompts:
        logger.info(f"  [{p['category']}] {p['prompt'][:100]}")
    return prompts
