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
    "base": (
        "Context is provided above. "
        "Read the context carefully and answer the question based on it.\n\n"
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
    )
}

# Answer: 제외

### Naive style
# "generate": f"\n\nGenerate a pseudo-passage which is relevant to the Question.",

### Query2Doc style
# "generate": (
#             "Few shot examples:\n\n"
#             "Write a passage that answers the given query:\n\n"
#             "Query: what state is this zip code 85282\n"
#             "Passage: Welcome to TEMPE, AZ 85282. 85282 is a rural zip code in Tempe, Arizona. The population"
#             "is primarily white, and mostly single. At $200,200 the average home value here is a bit higher than"
#             "average for the Phoenix-Mesa-Scottsdale metro area, so this probably isn’t the place to look for housing"
#             "bargains.5282 Zip code is located in the Mountain time zone at 33 degrees latitude (Fun Fact: this is the"
#             "same latitude as Damascus, Syria!) and -112 degrees longitude.\n\n"
#             "Query: why is gibbs model of reflection good\n"
#             "Passage: In this reflection, I am going to use Gibbs (1988) Reflective Cycle. This model is a recognised"
#             "framework for my reflection. Gibbs (1988) consists of six stages to complete one cycle which is able"
#             "to improve my nursing practice continuously and learning from the experience for better practice in the"
#             "future.n conclusion of my reflective assignment, I mention the model that I chose, Gibbs (1988) Reflective"
#             "Cycle as my framework of my reflective. I state the reasons why I am choosing the model as well as some"
#             "discussion on the important of doing reflection in nursing practice.\n\n"
#             "Query: what does a thousand pardons means\n"
#             "Passage: Oh, that’s all right, that’s all right, give us a rest; never mind about the direction, hang the"
#             "direction - I beg pardon, I beg a thousand pardons, I am not well to-day; pay no attention when I soliloquize,"
#             "it is an old habit, an old, bad habit, and hard to get rid of when one’s digestion is all disordered with eating"
#             "food that was raised forever and ever before he was born; good land! a man can’t keep his functions"
#             "regular on spring chickens thirteen hundred years old.\n\n"
#             "Query: what is a macro warning\n"
#             "Passage: Macro virus warning appears when no macros exist in the file in Word. When you open"
#             "a Microsoft Word 2002 document or template, you may receive the following macro virus warning,"
#             "even though the document or template does not contain macros: C:\<path>\<file name>contains macros."
#             "Macros may contain viruses.\n\n"
#             "Answer this Query by generating a relevant passage:\n\n"
#             ""
#             "Query: {question}\n"
#             "Passage: "
#         )

# Q : ~~ 
# A : ~~
# The answer to the question is A. Repeat the answer exactly
