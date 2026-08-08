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
        "Generate text completion prompts that test reading comprehension. "
        "Provide a short paragraph (2-4 sentences) describing a situation, then end with an incomplete sentence "
        "that requires understanding the paragraph to complete correctly. "
        "Example: 'The school garden had tomatoes, carrots, and sunflowers. Every Friday, students took turns "
        "watering the plants and pulling weeds. By the end of summer, the tomatoes were ripe and ready to pick. "
        "The students cared for the garden by'"
    ),
    "language_understanding": (
        "Generate text completion prompts that test language understanding. "
        "Write a vivid scene or set of instructions, then stop mid-sentence at a natural point. "
        "The model must understand context, tone, and intent to continue appropriately. "
        "Example: 'The cabin stood alone at the edge of the frozen lake, its windows glowing faintly "
        "through the snow. Inside, Mara hung her coat by the door, stamped the ice from her boots, and'"
    ),
    "world_knowledge": (
        "Generate text completion prompts that test factual world knowledge. "
        "Write prompts about science, geography, history, or general facts that require knowledge to complete. "
        "Use timeless facts, not recent events. "
        "Example: 'The process by which plants use sunlight to convert carbon dioxide and water into sugar is called'"
    ),
    "commonsense_reasoning": (
        "Generate text completion prompts that test commonsense reasoning. "
        "Describe an everyday situation and end with an incomplete sentence where common sense dictates the answer. "
        "Example: 'Jamal poured orange juice into a glass until it reached the top. When he tried to add more, the juice'"
    ),
    "language_modeling": (
        "Generate text completion prompts that test narrative language modeling. "
        "Write the beginning of a story or scene with rich detail, then stop mid-sentence. "
        "The model must continue in a coherent, natural way. "
        "Example: 'The old lighthouse keeper climbed the spiral stairs each evening, his lantern swaying with each step. "
        "At the top, he'"
    ),
    "causal_reasoning": (
        "Generate text completion prompts that test causal reasoning. "
        "Describe a cause and end with an incomplete sentence about its effect, or vice versa. "
        "Example: 'The driver forgot to turn off the headlights overnight, so in the morning the car battery'"
    ),
    "logical_inference": (
        "Generate text completion prompts that test logical inference. "
        "Present premises that lead to a conclusion the model must complete. "
        "Example: 'Every mammal breathes air. Whales are mammals. Therefore, whales'"
    ),
    "temporal_reasoning": (
        "Generate text completion prompts that test understanding of time and sequence. "
        "Describe events in order and ask the model to complete what happens next or what came before. "
        "Example: 'First she mixed the flour and eggs, then she added the sugar. After stirring for two minutes, she'"
    ),
}

SYSTEM_PROMPT = """You generate text completion prompts for evaluating language models.

Rules:
- Each prompt is an INCOMPLETE sentence or paragraph that the model must continue
- Write prompts as incomplete text, NOT questions
- The prompt must stop at a natural point where a good model would produce a meaningful continuation
- Vary length: some short (10-15 words), some longer (20-50 words)
- Prompts must be TIMELESS — use general knowledge, everyday scenarios, science facts, common sense
- Do NOT reference specific dates, recent events, or anything tied to a particular year
- Make prompts diverse — different topics, scenarios, styles
- Write engaging, varied prompts — avoid generic patterns like "X wanted to do Y" or "X and his friends did Y"
- Do NOT include the expected answer
- Make prompts CHALLENGING — they should require real knowledge, reasoning, or strong language skills to complete well. Avoid trivial completions that any model could produce

Return a JSON object with a "prompts" key containing an array of objects, each with a "prompt" field."""


def generate_prompts(eval_round, n_per_category=13):
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
