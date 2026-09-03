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
        "Generate prompts that test reading comprehension. "
        "Provide a short paragraph (2-4 sentences) describing a situation, then either end with an incomplete sentence or ask a question about the paragraph. "
        "Completion example: 'The school garden had tomatoes, carrots, and sunflowers. Every Friday, students took turns "
        "watering the plants and pulling weeds. By the end of summer, the tomatoes were ripe and ready to pick. "
        "The students cared for the garden by'\n"
        "Question example: 'A library charges $0.25 per day for overdue books. Maria returned her book 12 days late. How much does she owe and why might libraries use this system?'"
    ),
    "language_understanding": (
        "Generate prompts that test language understanding. "
        "Write a vivid scene or set of instructions, then either stop mid-sentence or ask a question about tone, intent, or meaning. "
        "Completion example: 'The cabin stood alone at the edge of the frozen lake, its windows glowing faintly "
        "through the snow. Inside, Mara hung her coat by the door, stamped the ice from her boots, and'\n"
        "Question example: 'A sign at a park reads: Dogs must be carried on the escalator. What does this actually mean, and why could it be misunderstood?'"
    ),
    "world_knowledge": (
        "Generate prompts that test factual world knowledge. "
        "Write prompts about science, geography, history, or general facts. Use timeless facts, not recent events. "
        "Completion example: 'The process by which plants use sunlight to convert carbon dioxide and water into sugar is called'\n"
        "Question example: 'Why do some metals rust when exposed to water while others do not?'"
    ),
    "commonsense_reasoning": (
        "Generate prompts that test commonsense reasoning. "
        "Describe an everyday situation and either end with an incomplete sentence or ask a question where common sense dictates the answer. "
        "Completion example: 'Jamal poured orange juice into a glass until it reached the top. When he tried to add more, the juice'\n"
        "Question example: 'If you leave an ice cube on a metal tray and another on a wooden board, which melts faster and why?'"
    ),
    "language_modeling": (
        "Generate prompts that test narrative language modeling. "
        "Write the beginning of a story or scene with rich detail, then either stop mid-sentence or ask a question about what happens next. "
        "Completion example: 'The old lighthouse keeper climbed the spiral stairs each evening, his lantern swaying with each step. "
        "At the top, he'\n"
        "Question example: 'A traveler arrives at a village where every door is painted red except one, which is black. What might this suggest about the village?'"
    ),
    "causal_reasoning": (
        "Generate prompts that test causal reasoning with CHAINS of causes and effects. "
        "Avoid simple single-cause questions. Require reasoning about multiple interacting factors, feedback loops, or non-obvious indirect effects. "
        "Completion example: 'When the dam upstream released extra water during heavy rains, the downstream farms flooded, which contaminated the wells, so the town'\n"
        "Question example: 'Why might removing wolves from a national park eventually lead to riverbanks eroding faster?'"
    ),
    "logical_inference": (
        "Generate prompts that test logical inference. "
        "Present premises and either end with an incomplete conclusion or ask the model to draw one. "
        "Completion example: 'Every mammal breathes air. Whales are mammals. Therefore, whales'\n"
        "Question example: 'All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?'"
    ),
    "temporal_reasoning": (
        "Generate prompts that test understanding of time and sequence. "
        "Describe events in order and either end with an incomplete sentence or ask about the sequence. "
        "Completion example: 'First she mixed the flour and eggs, then she added the sugar. After stirring for two minutes, she'\n"
        "Question example: 'Tom woke up, ate breakfast, then realized he had forgotten to set his alarm. In what order did these events actually happen?'"
    ),
    "math_reasoning": (
        "Generate prompts that test mathematical reasoning. "
        "Require MULTI-STEP reasoning: combining operations, unit conversions, proportions, or logical deduction. "
        "Avoid single-operation problems like basic division or addition. Each problem should need at least 2-3 steps to solve. "
        "Completion example: 'A store discounts a $80 jacket by 25%, then adds 10% sales tax to the discounted price. The final price is'\n"
        "Question example: 'A tank fills at 3 liters per minute but leaks at 0.5 liters per minute. If it starts half-full at 50 liters capacity, how long until it overflows?'"
    ),
    "truthfulness": (
        "Generate prompts that test whether a model avoids common misconceptions and myths. "
        "Write about a popular misconception and either end with an incomplete sentence or ask the model to evaluate the claim. "
        "Completion example: 'Despite what many people believe, the Great Wall of China is actually not visible from space because it is'\n"
        "Question example: 'Is it true that humans only use 10% of their brain? Explain why or why not.'"
    ),
    "pronoun_resolution": (
        "Generate prompts that test pronoun and coreference resolution. "
        "Write a sentence with an ambiguous pronoun and either end mid-sentence or ask what the pronoun refers to. "
        "Completion example: 'The bottle did not fit in the suitcase because it was too large, so they decided to'\n"
        "Question example: 'The teacher told the student that she needed to improve. Who needs to improve, and how can you tell?'"
    ),
    "paraphrase_detection": (
        "Generate prompts that test whether two statements mean the same thing. "
        "Present two sentences and either end with an incomplete comparison or ask whether they convey the same meaning. "
        "Completion example: 'Statement A: The cat chased the mouse. Statement B: The mouse was pursued by the cat. These two statements'\n"
        "Question example: 'Do these two sentences mean the same thing? \"She failed to avoid the obstacle\" and \"She hit the obstacle.\"'"
    ),
    "word_sense": (
        "Generate prompts that test word sense disambiguation in TRICKY contexts. "
        "Use sentences where the word's meaning is genuinely ambiguous or where the obvious interpretation is wrong. "
        "Avoid cases where context makes the meaning immediately obvious. The reader should need to think carefully. "
        "Completion example: 'The manager said the pitch needed more polish before the board would consider it. Whether pitch refers to a sales presentation or a playing field changes the meaning entirely, and the clue is'\n"
        "Question example: 'In \"The doctor told her she had a rare condition and should avoid drafts,\" does drafts mean air currents, written documents, or preliminary versions? What makes this ambiguous?'"
    ),
}

SYSTEM_PROMPT = """You generate prompts for evaluating language models. Two formats:

1. COMPLETION — an incomplete sentence or paragraph the model must continue
2. QUESTION — a direct question the model must answer

Aim for roughly 70% questions and 30% completions within each batch.

Rules:
- Vary length: some short (10-15 words), some longer (20-50 words)
- Prompts must be TIMELESS — use general knowledge, everyday scenarios, science facts, common sense
- Do NOT reference specific dates, recent events, or anything tied to a particular year
- Make prompts diverse — different topics, scenarios, styles
- Write engaging, varied prompts — avoid generic patterns like "X wanted to do Y" or "X and his friends did Y"
- Do NOT include the expected answer
- Make prompts CHALLENGING — they should require real knowledge, reasoning, or strong language skills to answer or complete well. Avoid trivial prompts that any model could produce
- For completions: stop at a natural point where a good model would produce a meaningful continuation. Avoid prompts where the answer is a single obvious word. Prefer prompts that can be completed in multiple valid ways
- For questions: avoid questions with a single well-known answer. Prefer questions that require reasoning across multiple concepts, comparing trade-offs, or explaining WHY something works rather than just WHAT it is. Bad: "What is photosynthesis?" Good: "Why do plants at the bottom of a rainforest canopy have larger leaves than those at the top?"

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
