ALL_PROMPTS = {
    "base": f"\n\nRepeat the previous Answer exactly.",
    "query-dependent": f"\n\nThe answer to the question is A. Repeat the answer exactly.",
    "query-dependent-only": f"\n\nThe answer to the question is A.",
    "context-prefilling-base": f"\n\nRepeat the previous context exactly."
}

PSEUDO_PASSAGE_PROMPT = {
    "pseudo-passage-base": {
        "generate": (
            "Generate a document that provides accurate and relevant information to answer the given question.\n\n"
            "Question: {question} Document:"
        ),
        "repeat": f"\n\nRepeat the previous context exactly."
    }
}

GENERATE_PROMPT = {
    # "base": (
    #     "Context is provided above. "
    #     "Read the context carefully and answer the question based on it.\n\n"
    #     "Question: {question}\n\n"
    #     "Answer: "
    # ),
    "base": (
        "Context is provided above. "
        "Read the context carefully and answer the question based on it briefly.\n\n"
        "Question: {question}\n\n"
        "Answer: "
    ),
    "pure-llm": (
        "Answer the question based on your knowledge.\n\n"
        "Question: {question}\n\n"
        "Answer: "
    ),
    "pure-llm-brief": (
        "Answer the question based on your knowledge briefly. "
        "Be concise and direct.\n\n"
        "Question: {question}\n\n"
        "Answer: "
    ),
    "pure-llm-brief-2": (
        "Answer the question based on your knowledge briefly. "
        "Output only the answer entity or phrase. "
        "Do not use complete sentences.\n\n"
        "Question: {question}\n\n"
    ),
    "mj_prompt_v2": (
        "Answer the Question\n\n"
        "Question: {question}\n\n"
    ),
    "care_closed_book": (
        "Answer the questions:\n"
        "Question: {question}?\n"
        "The answer is:"
    ),
    "priori_judgement": (
        "Answer the following question based on your internal knowledge with one or few words.\n"
        "Question: {question}\n"
        "Answer: "
    )
}