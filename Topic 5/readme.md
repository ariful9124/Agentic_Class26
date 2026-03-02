# **Exercise 0:** Set-up — Get notebook running; unzip Corpora.zip. Use PDFs from `Corpora/<corpus>/pdf_embedded/`.
Answer: Completed

# Exercise 1: Question:
- Does the model hallucinate specific values without RAG?
- Does RAG ground the answers in the actual manual?
- Are there questions where the model's general knowledge is actually correct?
Answer: 

- Hallucination (no RAG): The model consistently invented specific values — a wrong spark plug gap (0.5mm, then self-correcting to 3.175mm), a fabricated oil grade (10W-30), and entirely made-up Congressional events (e.g., Flood doubting a bill, Stefanik voting against her own party's budget). All January 2026 content was pure confabulation since it's past the training cutoff.

- RAG grounding: Strong for well-indexed content — the transmission band fix and Mayor Black recognition were answered accurately with page-level citations. The weak spot was the oil question: retrieval scores were low (0.37–0.40) and the model supplemented with brand recommendations (Mobil 1, Castrol) instead of admitting the corpus lacked the answer. The Stefanik question failed similarly even with RAG.

- Without RAG, the model performs best on general mechanical principles and widely known technical concepts, such as how carburetors function or how mechanical adjustments typically work. It performs poorly on historical specifications, exact numeric values, domain-specific procedures, and recent events, where retrieval-based grounding is necessary for accuracy.

Optional subtask: Put both the Model T manual and the Congressional Record issues into the same RAG database.  Does this have any affect on the quality of the answers?
Answer: 
Combining the Model T manual (~632 chunks) and the Congressional Record (~47,878 chunks) into a single RAG database (~48,510 chunks total) had minimal impact on overall answer quality, as retrieval generally remained domain-appropriate: Model T questions still retrieved manual passages and Congressional questions retrieved legislative records. The RAG-based answers were similar in accuracy to those obtained from separate databases, indicating that semantic retrieval was strong enough to distinguish between domains despite the large imbalance in corpus size. However, the combined index introduced minor retrieval noise, such as occasional irrelevant chunks (e.g., Congressional passages appearing in Model T oil queries) with lower similarity scores. These did not usually affect final answers but indicate mild cross-domain interference. Overall, merging the corpora did not significantly degrade performance, though domain-specific indexing or filtering would likely improve robustness for ambiguous queries.

# Exercise 2:

1. Does GPT-4o Mini do a better job than Qwen 2.5 1.5B in avoiding hallucinations?
Yes. GPT-4o Mini consistently avoided hallucinations better than Qwen 2.5 1.5B. For the Congressional Record questions (2026 events), GPT-4o Mini correctly stated that the information was beyond its training cutoff instead of inventing facts. In contrast, Qwen 2.5 1.5B frequently fabricated details (e.g., Stefanik voting records, infrastructure bills, or incorrect policy claims) when run without RAG. Even for Model T questions, GPT-4o Mini produced plausible general knowledge answers, while Qwen often generated confident but incorrect specifics such as wrong spark plug gaps and modern oil grades.

2. Which questions does GPT-4o Mini answer correctly? Compare the training cutoff and corpus age.
GPT-4o Mini answered general mechanical questions about the Model T (Q1–Q4) directionally correctly, because these topics are widely documented and predate its October 2023 training cutoff by over a century (Model T manual from 1919). However, it could not answer Congressional Record questions (Q5–Q8) that refer to January 2026 events, since these occur well after its training period. This shows that GPT-4o Mini’s parametric knowledge is reliable for older historical domains like the Model T, but recent documents such as the 2026 Congressional Record require RAG for accurate answers.

# Exercise 3:
