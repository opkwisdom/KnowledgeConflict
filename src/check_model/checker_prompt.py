OPENAI = {
    "base": {
        "system": (
            "You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.\n"
            "Your task is to judge the correctness of an answer based on a query and provided contexts.\n"
            "You must also identify which contexts were used, contradictory, or irrelevant."
        ),
        "user": (
            "### Task\n"
            "1. Determine if the Internal Answer is factually correct relative to the Query.\n"
            "2. Identify the indices of contexts that support (positive), contradict (negative), or are irrelevant to the answer.\n\n"
            "### Input Data\n"
            "- Query: {query}\n"
            "- Internal Answer: {internal_answer}\n\n"
            "### Contexts\n"
            "{formatted_contexts}\n\n"
            "Provide your assessment in the structured format."
        )
    }
}