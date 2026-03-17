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
    "sce_modified": {
        "system": (
            "You are a Generous Evaluator for a QA system. "
            "Your goal is to validate the model's knowledge while strictly filtering context utility. "
            "Favor the Internal Answer if it is factually grounded."
        ),
        "user": (
            "### Task Overview\n"
            "Evaluate the 'Internal Answer' and the single provided 'Context'.\n\n"

            "### Step 1: Verify Internal Answer (Parametric Knowledge)\n"
            "Does the 'Internal Answer' represent the correct entity or meaning for the 'Query'?\n"
            "   - **is_correct**: Set to `true` if the answer is factually correct.\n"
            # "   - **Guideline**: Do NOT be pedantic. If the core fact is right (e.g., 'Jobs' instead of 'Steve Jobs'), accept it as `true`.\n\n"

            "### Step 2: Classify Context Utility \n"
            "Only mark the context as useful if it is undeniable.\n"
            "   - **Positive**: The context contains the answer explicitly and unambiguously.\n"
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
    ### Judge prompts without internal answer
    "judge_only_contexts": {
        "system": (
            "You are an extremely strict evaluator for a Retrieval-Augmented Generation (RAG) system.\n"
            "Judge whether a provided Context contains sufficient information to answer a given Query.\n"
        ),
        "user": (
            "### Task Description\n"
            "Classifiy EACH context (from [0] to [{last_index}]) into one of three categories based on the following rules. "
            "DO NOT skip any context:\n\n"
            "1. **Positive**: The context contains the specific information to derive the answer literally. "
            "Deduction or general knowledge is NOT allowed. The evidence must be present in the text.\n"
            "2. **Negative (Broad Definition)**: Classify as Negative if the context shares ANY semantic relationship, keywords, or topic with the Query, "
            "but it FAILS to provide the answer. This includes 'hard negatives', 'partial information' or 'outdated facts'.\n"
            "3. **Irrelevant**: The context is unrelated to the Query, discusses a different entity, or is completely off-topic.\n\n"

            "### Input\n"
            "- Query: {query}\n"
            "- Contexts:\n{formatted_contexts}\n\n"
            "Provide the output strictly using the provided JSON schema."
        )
    },
    "judge_only_contexts_sci": {
        "system": (
            "You are a strict evaluator for a Knowledge Conflict Resolution system.\n"
            "Evaluate the logical relationship between the provided Context and a Reference Answer regarding a specific Query.\n"
        ),
        "user": (
            "### Task Description\n"
            "Classify EACH context (from [0] to [{last_index}]) into one of three strict categories based on the following rules. "
            "DO NOT skip any context:\n\n"
            "1. **Supportive (S)**: The context contains the exact facts or explicit evidence that perfectly aligns with the Reference Answer.\n"
            "2. **Contradictory (C)**: The context directly opposes, denies, or provides mutually exclusive factual information against the Reference Answer. It MUST actively claim a different truth.\n"
            "3. **Irrelevant (I)**: The context fails to provide a direct answer, only shares superficial keywords, or is completely off-topic.\n\n"

            "### Input\n"
            "- Query: {query}\n"
            "- Reference Answer: {true_answer}\n"
            "- Contexts:\n{formatted_contexts}\n\n"
            "Provide the output strictly using the provided JSON schema."
        )
    },
    "judge_only_contexts_sci_reasoning": {
        "system": (
            "You are a strict evaluator for a Knowledge Conflict Resolution system.\n"
            "Evaluate the logical relationship between the provided Context and a Reference Answer regarding a specific Query.\n"
        ),
        "user": (
            "### Task Description\n"
            "Classify EACH context (from [0] to [{last_index}]) into one of three strict categories based on the following rules. "
            "DO NOT skip any context:\n\n"
            "1. **Supportive (S)**: The context contains facts or evidence that support or align with the Reference Answer.\n"
            "2. **Contradictory (C)**: The context provides factual information that is mutually exclusive to, or logically conflicts with the Reference Answer.\n"
            "   *(CRITICAL: The context MUST be about the EXACT SAME entity/topic as the Query. If the query asks about 'Season 12' and the context describes 'Season 13', it is NOT a contradiction, but Irrelevant.)*\n"
            "3. **Irrelevant (I)**: The context fails to provide a direct answer, only shares superficial keywords, or is off-topic.\n\n"

            "### Special Instruction for Reasoning\n"
            "When writing your `reasoning`, if the Reference Answer or Context contains dates, years, or centuries, you MUST explicitly convert "
            "and compare their actual year ranges (e.g., \"16th century = 1501-1600, which does not match 1607\") before classifying."

            "### Input\n"
            "- Query: {query}\n"
            "- Reference Answer: {ref_answer}\n"
            "- Contexts:\n{formatted_contexts}\n\n"
            "Provide the output strictly using the provided JSON schema. Crucially, you MUST first use the reasoning field "
            "to write down a step-by-step logical explanation of why each context belongs to S, C, or I before assigning their indices."
        )
    },
    "judge_few_shot_anchor_v2": {
        "system": (
            "You are a strict evaluator for a Knowledge Conflict Resolution system.\n"
            "Evaluate the logical relationship between the provided Context and a Reference Answer regarding a specific Query. "
            "You MUST follow a strict step-by-step logical gating process."
        ),
        "user": (
            "### Step-by-Step Classification Rules\n"
            "Classify EACH context (from [0] to [{last_index}]) into one of three strict categories based on the following rules. "
            "DO NOT skip any context:\n\n"
            "**Step 1: Entity & Topic Match (The Gating Rule)**\n"
            "Does the context discuss the EXACT SAME entity, event, and temporal scope as the Query and Reference Answer?\n"
            "- If No: Stop and Classify as **Irrelevant (I)**. (e.g., Query is about 'Season 12', context discusses 'Season 13').\n"
            "- If Yes: Move to Step 2.\n\n"
            "**Step 2: Fact Check & Alignment**\n"
            "Compare the specific facts in the context against the Reference Answer.\n"
            "- If the context provides facts that are mutually exclusive to or logically conflict with the Reference Answer: Classify as **Contradictory (C)**.\n"
            "- If the context contains facts or evidence that support or align with the Reference Answer: Classify as **Supportive (S)**.\n"
            "- If the context fails to provide a direct answer to the query, or only shares superficial keywords: Classify as **Irrelevant (I)**.\n\n"
            
            "### Anchor Examples (STUDY THESE CAREFULLY)\n"
            "You MUST output your final answer as a JSON object with an 'evaluations' list. Follow this structure:\n\n"
            
            "**[Example Scenarios]**\n"
            "- Query: When is the finale of season 12?\n"
            "- Reference Answer: Sept 18.\n"
            "- Contexts:\n"
            "[0] Season 13 finale was on Sept 19.\n"
            "[1] The 12th season ended on September 18th.\n"
            "[2] The 12th season finale aired on October 1st.\n"
            "[3] The show was renewed for a 12th season.\n\n"
            
            "### Input\n"
            "- Query: {query}\n"
            "- Reference Answer: {ref_answer}\n"
            "- Contexts:\n{formatted_contexts}\n\n"
            "Analyze ALL contexts from [0] to [{last_index}] using the Step-by-Step rules. Return ONLY a valid JSON object matching the expected schema."
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
    },
    "judge_only_contexts_sci": {
        "system": (
            "You are a strict evaluator for a Knowledge Conflict Resolution system.\n"
            "Evaluate the logical relationship between the provided Context and a Reference Answer regarding a specific Query.\n"
        ),
        "user": (
            "### Task Description\n"
            "Classify EACH context (from [0] to [{last_index}]) into one of three strict categories based on the following rules. "
            "DO NOT skip any context:\n\n"
            "1. **Supportive (S)**: The context contains facts or evidence that support or align with the Reference Answer.\n"
            "2. **Contradictory (C)**: The context provides factual information that is mutually exclusive to, or logically conflicts with the Reference Answer.\n"
            "3. **Irrelevant (I)**: The context fails to provide a direct answer, only shares superficial keywords, or is off-topic.\n\n"

            "### Input\n"
            "- Query: {query}\n"
            "- Reference Answer: {ref_answer}\n"
            "- Contexts:\n{formatted_contexts}\n\n"
            "Provide the output strictly using the provided JSON schema."
        )
    },
    "judge_only_contexts_sci_reasoning": {
        "system": (
            "You are a strict evaluator for a Knowledge Conflict Resolution system.\n"
            "Evaluate the logical relationship between the provided Context and a Reference Answer regarding a specific Query.\n"
        ),
        "user": (
            "### Task Description\n"
            "Classify EACH context (from [0] to [{last_index}]) into one of three strict categories based on the following rules. "
            "DO NOT skip any context:\n\n"
            "1. **Supportive (S)**: The context contains facts or evidence that support or align with the Reference Answer.\n"
            "2. **Contradictory (C)**: The context provides factual information that is mutually exclusive to, or logically conflicts with the Reference Answer.\n"
            "   *(CRITICAL: The context MUST be about the EXACT SAME entity/topic as the Query. If the query asks about 'Season 12' and the context describes 'Season 13', it is NOT a contradiction, but Irrelevant.)*\n"
            "3. **Irrelevant (I)**: The context fails to provide a direct answer, only shares superficial keywords, or is off-topic.\n\n"

            "### Special Instruction for Reasoning\n"
            "When writing your `reasoning`, if the Reference Answer or Context contains dates, years, or centuries, you MUST explicitly convert "
            "and compare their actual year ranges (e.g., \"16th century = 1501-1600, which does not match 1607\") before classifying."

            "### Input\n"
            "- Query: {query}\n"
            "- Reference Answer: {ref_answer}\n"
            "- Contexts:\n{formatted_contexts}\n\n"
            "Provide the output strictly using the provided JSON schema. Crucially, you MUST first use the reasoning field "
            "to write down a step-by-step logical explanation of why each context belongs to S, C, or I before assigning their indices."
        )
    },
    "judge_few_shot_anchor": {
        "system": (
            "You are a strict evaluator for a Knowledge Conflict Resolution system.\n"
            "Evaluate the logical relationship between the provided Context and a Reference Answer regarding a specific Query.\n"
        ),
        "user": (
            "### Classification Rules\n"
            "Classify EACH context (from [0] to [{last_index}]) into one of three strict categories based on the following rules. "
            "DO NOT skip any context:\n\n"
            "1. **Supportive (S)**: The context contains facts or evidence that support or align with the Reference Answer.\n"
            "2. **Contradictory (C)**: The context provides factual information that is mutually exclusive to, or logically conflicts with the Reference Answer.\n"
            "   *(CRITICAL: The context MUST be about the EXACT SAME entity/topic as the Query. If the query asks about 'Season 12' and the context describes 'Season 13', it is NOT a contradiction, but Irrelevant.)*\n"
            "3. **Irrelevant (I)**: The context fails to provide a direct answer, only shares superficial keywords, or is off-topic.\n\n"
            
            "### Anchor Examples (STUDY THESE CAREFULLY)\n"
            
            "**[Example A - Entity Mismatch]**\n"
            "- Query: When is the finale of season 12?\n"
            "- Reference Answer: Sept 18.\n"
            "- Context: Season 13 finale was on Sept 19.\n"
            "-> Reasoning: The context discusses Season 13, not 12. Different entity.\n"
            "Classification: I.\n\n"
            
            "**[Example B - Temporal Conflict]**\n"
            "- Query: When did the first colony start?\n"
            "- Reference Answer: 16th century.\n"
            "- Context: Jamestown started in 1607.\n"
            "-> Reasoning: 1607 is the 17th century. The reference claims 16th. These are mutually exclusive for the same event.\n"
            "Classification: C.\n\n"
            
            "**[Example 3 - Fact Alignment]**\n"
            "- Query: Who wrote Hamlet?\n"
            "- Reference Answer: William Shakespeare.\n"
            "- Context: Hamlet is a tragedy written by English playwright William Shakespeare in the early 1600s.\n"
            "-> Reasoning: The context explicitly states Shakespeare wrote Hamlet, which perfectly aligns with the reference answer.\n"
            "Classification: S.\n\n"
            
            "### Input\n"
            "- Query: {query}\n"
            "- Reference Answer: {ref_answer}\n"
            "- Contexts:\n{formatted_contexts}\n\n"
            "Provide your step-by-step reasoning first, mirroring the Anchor Examples logic, then assign the category."
        )
    },
    "judge_few_shot_anchor_v2": {
        "system": (
            "You are a strict evaluator for a Knowledge Conflict Resolution system.\n"
            "Evaluate the logical relationship between the provided Context and a Reference Answer regarding a specific Query. "
            "You MUST follow a strict step-by-step logical gating process."
        ),
        "user": (
            "### Step-by-Step Classification Rules\n"
            "Classify EACH context (from [0] to [{last_index}]) into one of three strict categories based on the following rules. "
            "DO NOT skip any context:\n\n"
            "**Step 1: Entity & Topic Match (The Gating Rule)**\n"
            "Does the context discuss the EXACT SAME entity, event, and temporal scope as the Query and Reference Answer?\n"
            "- If No: Stop and Classify as **Irrelevant (I)**. (e.g., Query is about 'Season 12', context discusses 'Season 13').\n"
            "- If Yes: Move to Step 2.\n\n"
            "**Step 2: Fact Check & Alignment**\n"
            "Compare the specific facts in the context against the Reference Answer.\n"
            "- If the context provides facts that are mutually exclusive to or logically conflict with the Reference Answer: Classify as **Contradictory (C)**.\n"
            "- If the context contains facts or evidence that support or align with the Reference Answer: Classify as **Supportive (S)**.\n"
            "- If the context fails to provide a direct answer to the query, or only shares superficial keywords: Classify as **Irrelevant (I)**.\n\n"
            
            "### Anchor Examples (STUDY THESE CAREFULLY)\n"
            "You MUST output your final answer as a JSON object with an 'evaluations' list. Follow this structure:\n\n"
            
            "**[Example Scenarios]**\n"
            "- Query: When is the finale of season 12?\n"
            "- Reference Answer: Sept 18.\n"
            "- Contexts:\n"
            "[0] Season 13 finale was on Sept 19.\n"
            "[1] The 12th season ended on September 18th.\n"
            "[2] The 12th season finale aired on October 1st.\n"
            "[3] The show was renewed for a 12th season.\n\n"
            
            "-> Expected JSON Output:\n"
            "{{\n"
            "  \"evaluations\": [\n"
            "    {{\"index\": 0, \"reasoning\": \"Step 1 Fails: Discusses Season 13, not 12. Different entity.\", \"category\": \"I\"}},\n"
            "    {{\"index\": 1, \"reasoning\": \"Step 1 Passes. Step 2: Explicitly states season 12 ended on Sept 18, matching reference.\", \"category\": \"S\"}},\n"
            "    {{\"index\": 2, \"reasoning\": \"Step 1 Passes. Step 2: Oct 1st contradicts the reference date of Sept 18.\", \"category\": \"C\"}},\n"
            "    {{\"index\": 3, \"reasoning\": \"Step 1 Passes. Step 2: Mentions season 12 but gives no air date. Superficial.\", \"category\": \"I\"}}\n"
            "  ]\n"
            "}}\n\n"
            
            "### Input\n"
            "- Query: {query}\n"
            "- Reference Answer: {ref_answer}\n"
            "- Contexts:\n{formatted_contexts}\n\n"
            "Analyze ALL contexts from [0] to [{last_index}] using the Step-by-Step rules. Return ONLY a valid JSON object matching the expected schema."
        )
    }
}