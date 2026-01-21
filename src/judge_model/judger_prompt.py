from .template import apply_template

OPENAI = {
    "base": {
        "system": (
            "You are an extremely strict evaluator for a Retrieval-Augmented Generation (RAG) system.\n"
            "Your task is to classify the relevance of retrieved contexts according to specific types"
            "and to judge the factual correctness of the answer."
        ),
        "user": (
            "### Task Description\n"
            "1. **Reasoning**: First, briefly explain your reasoning for classifying the contexts and evaluating the answer.\n\n"

            "2. **Analyze Contexts**: Classify EVERY single context (from [0] to [{last_index}]) into one of three categories based on these rules"
            "DO NOT skip any context:\n\n"
            "   - **Positive**: The context contains the specific information to derive the answer literally. "
            "Deduction or general knowledge is NOT allowed. The evidence must be present in the text.\n"
            "   - **Negative (Broad Definition)**: Classify as Negative if the context shares ANY semantic relationship, keywords, or topic with the Query, "
            "but it FAILS to provide the answer. This includes 'hard negatives', 'partial information' or 'outdated facts'.\n"
            "   - **Irrelevant**: The context is unrelated to the Query, discusses a different entity, or is completely off-topic.\n\n"

            "3. **Determine Correctness**: Determine if the 'Internal Answer' provides a factually correct response to the 'Query'.\n\n"
            

            "### Input Data\n"
            "- Query: {query}\n"
            "- Internal Answer: {internal_answer}\n\n"
            "### Contexts\n"
            "{formatted_contexts}\n\n"
            "Provide the output strictly using the provided JSON schema."
        )
    },
    "single_eval": {
        "system": (
            "You are an objective evaluator for a Retrieval-Augmented Generation (RAG) system.\n"
            "Your task is to classify the relevance of a single retrieved context and to judge the factual correctness of the answer."
        ),
        "user": (
            "### Task Description\n"
            "1. **Analyze Contexts**: Classify the provided context (index 0) as 'positive' or 'negative'.\n"
            "   - **Positive**: The context provides sufficient information or evidence to derive the answer. "
            "Direct mention is preferred, but clear semantic matches and obvious implications are also allowed.\n"
            "   - **Negative**: The context lacks the necessary information, provides only partial clues,"
            " contains outdated facts, or is entirely off-topic.\n\n"

            "2. **Determine Correctness**: Determine if the 'Internal Answer' provides a factually correct response to the 'Query'.\n\n"
            
            "### One-Shot Examples\n"
            "**Example 1: Positive**\n"
            "- Query: where did they film hot hub time machine?\n"
            "- Internal Answer: Fernie Alpine Resort\n"
            "- Context: [0]\nTitle: Hot Tub Time Machine\n\n...It was filmed primarily at the Vancouver Film Studios"
            " in Vancouver and the Fernie Alpine Resort in Fernie, British Columbia.\n"
            "- Output: {{ \"is_correct\": true, \"ctx_relevance\": {{ \"positive\": [0], \"negative\": [], \"irrelevant\": [] }} }}\n\n"

            "**Example 2: Negative**\n"
            "- Query: where did they film hot tub time machine\n"
            "- Internal Answer: Fernie Alpine Resort\n"
            "- Context: [0]\nTitle: Fernie Alpine Resort\n\nDuring spring 2009, Fernie Alpine Resort was transformed into the fictional Kodiak Valley ski resort, circa 1986, for exterior location shots of the Hollywood\n"
            "- Output: {{ \"is_correct\": true, \"ctx_relevance\": {{ \"positive\": [], \"negative\": [0], \"irrelevant\": [] }} }}\n\n"

            "### Input Data\n"
            "- Query: {query}\n"
            "- Internal Answer: {internal_answer}\n"
            "- Context: {formatted_contexts}\n\n"

            "### Output Rules (Strict)\n"
            "1. Return a valid JSON object matching the JudgeOutput schema:\n"
            "2. You MUST use index [0] for the classification.\n"
            "- 'is_correct': boolean\n"
            "- 'ctx_relevance': {{ 'positive': [0], 'negative': [], 'irrelevant': [] }} (if Positive)\n"
            "- 'ctx_relevance': {{ 'positive': [], 'negative': [0], 'irrelevant': [] }} (if Negative)\n"
        )
    },
    "mj_prompt": {
        "system": (
            "You are an expert evaluator for a RAG system. "
            "Your objective is to assess the relevance of retrieved passages and the factual accuracy of the answer."
        ),
        "user": (
            "### Task Description\n"
            "You must evaluate the provided 'Contexts' and the 'Internal Answer' based on the 'Query'.\n\n"

            "### Step 1: Classify Context Relevance\n"
            "Analyze EACH context (from [0] to [{last_index}]) and assign one of the following labels. "
            "Be precise in distinguishing 'Negative' from 'Irrelevant'.\n\n"

            "   - **Positive**: The context contains sufficient information to answer the query. "
            "Direct evidence or strong clues allowing logical deduction are present.\n"
            "   - **Negative** (Targeted Hard Negative): The context focuses on the EXACT SAME entity or event as the query "
            "but FAILS to provide the specific answer (e.g., Query asks for 'release date', Context only gives 'director'). "
            "If it just shares keywords but talks about a different sub-topic, mark it as Irrelevant.\n"
            "   - **Irrelevant**: The context is about a different entity, a different time period, or is generally unrelated, "
            "even if it shares some keywords.\n\n"

            "### Step 2: Judge Answer Correctness\n"
            "Determine if the 'Internal Answer' is factually correct based on the 'Query'.\n"
            "   - **Ignore Style**: Even if the answer is verbose or grammatically imperfect, if it contains the correct core entity/fact, mark it as **True**.\n"
            "   - **Fact Check**: If the core entity/number/date is wrong, mark it as **False**.\n\n"

            "### Input Data\n"
            "- Query: {query}\n"
            "- Internal Answer: {internal_answer}\n\n"
            "### Contexts\n"
            "{formatted_contexts}\n\n"

            "### Output Format\n"
            "Return the result strictly in the following JSON format:\n"
            "{{\n"
            "  \"reasoning\": \"Briefly explain why contexts are Positive/Negative/Irrelevant and why the answer is Correct/Incorrect.\",\n"
            "  \"ctx_relevance\": [\"Positive\", \"Irrelevant\", \"Negative\", ...],\n"
            "  \"is_correct\": true\n"
            "}}"
        )
    },
    "single_context_eval": {
        "system": (
            "You are a Balanced Evaluator for a QA system. "
            "Your goal is to validate the model's knowledge while strictly filtering context utility. "
            "Favor the Internal Answer if it is factually grounded."
        ),
        "user": (
            "### Task Overview\n"
            "Evaluate the 'Internal Answer' and the single provided 'Context'.\n\n"

            "### Step 1: Verify Internal Answer (Parametric Knowledge)\n"
            "Does the 'Internal Answer' represent the correct entity or meaning for the 'Query'?\n"
            "   - **is_correct**: Set to `true` if the answer is factually correct.\n"
            "   - **Guideline**: Do NOT be pedantic. If the core fact is right (e.g., 'Jobs' instead of 'Steve Jobs'), accept it as `true`.\n\n"

            "### Step 2: Classify Context Utility \n"
            "Only mark the context as useful if it is undeniable.\n"
            "   - **Positive**: The context contains the answer **EXPLICITLY** and **UNAMBIGUOUSLY**.\n"
            "   - **Negative**: The context discusses the topic but does not contain the answer, or requires excessive guessing.\n\n"

            "### Input Data\n"
            "- Query: {query}\n"
            "- Internal Answer: {internal_answer}\n"
            "- Contexts:\n{formatted_contexts}\n\n"

            "### Output Requirement\n"
            "Return valid JSON strictly adhering to the schema.\n"
            "**Judge based on actual input.**\n"
            "Example (Correct Internal Answer):\n"
            "{{ \"is_correct\": true, \"ctx_relevance\": {{ \"positive\": [], \"negative\": [] }} }}"
            "{{ \"is_correct\": false, \"ctx_relevance\": {{ \"positive\": [0], \"negative\": [] }} }}"
            "{{ \"is_correct\": false, \"ctx_relevance\": {{ \"positive\": [], \"negative\": [0] }} }}"
        )
    },
    "single_context_eval_2": {
        "system": (
            "You are a Balanced Evaluator for a QA system. "
            "Your goal is to validate the model's knowledge while strictly filtering context utility. "
            "Favor the Internal Answer if it is factually grounded."
        ),
        "user": (
            "### Task Overview\n"
            "Evaluate the 'Internal Answer' and the single provided 'Context'.\n\n"

            "### Step 1: Verify Internal Answer (Parametric Knowledge)\n"
            "Does the 'Internal Answer' represent the correct entity or meaning for the 'Query'?\n"
            "   - **is_correct**: Set to `true` if the answer is factually correct.\n"
            "   - **Guideline**: Do NOT be pedantic. If the core fact is right (e.g., 'Jobs' instead of 'Steve Jobs'), accept it as `true`.\n\n"

            "### Step 2: Classify Context Utility \n"
            "Only mark the context as useful if it is undeniable.\n"
            "   - **Positive**: The context contains the answer **EXPLICITLY** and **UNAMBIGUOUSLY**.\n"
            "   - **Negative**: The context discusses the topic but does not contain the answer, or requires excessive guessing.\n\n"

            "### Input Data\n"
            "- Query: {query}\n"
            "- Internal Answer: {internal_answer}\n"
            "- Contexts:\n{formatted_contexts}\n\n"

            "### Output Requirement\n"
            "Return valid JSON strictly adhering to the schema.\n"
            "**Judge based on actual input.**\n"
            "Example (Correct Internal Answer):\n"
            "{{ \"is_correct\": true, \"ctx_relevance\": {{ \"positive\": [], \"negative\": [] }} }}"
            "{{ \"is_correct\": false, \"ctx_relevance\": {{ \"positive\": [0], \"negative\": [] }} }}"
            "{{ \"is_correct\": false, \"ctx_relevance\": {{ \"positive\": [], \"negative\": [0] }} }}"
        )
    }
}


HUGGINGFACE = {
    "base": {
        # template.py가 "You are a helpful assistant." 뒤에 붙일 내용
        "system": (
            "You are an extremely strict evaluator for a RAG system. "
            "Your goal is to evaluate context relevance and answer correctness based on strict guidelines."
        ),
        
        # apply_template의 'query' 인자로 들어갈 내용
        "user": (
            "### Task Description\n"
            "1. **Reasoning**: Explain your logic briefly.\n"
            "2. **Analyze Contexts**: Classify EVERY context (from [0] to [{last_index}]) into one of three categories based on these rules"
            "DO NOT skip any context:\n\n"
            "   - **Positive**: Contains specific, literal evidence for the answer. No deduction allowed.\n"
            "   - **Negative**: Topic/keywords match but FAILS to answer (Hard Negative/Distractor).\n"
            "   - **Irrelevant**: Off-topic or unrelated.\n"
            "3. **Determine Correctness**: Check if the Internal Answer provides a factually correct response to the Query.\n\n"

            "### Input Data\n"
            "- Query: {query}\n"
            "- Internal Answer: {internal_answer}\n\n"
            
            "### Contexts\n"
            "{formatted_contexts}\n\n"
            
            "Analyze the data and provide the structured evaluation."
        )
    }
}