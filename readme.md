## Task 1: Completed. Added functions to handle 'verbose' 

#📌 Note: Why the Agent Appears to Take Multiple Turns

When the user inputs:
```
hi
```
the program sends **a single prompt** to the LLM:
```
User: hi
Assistant:
```

The LLM then generates text **continuously** until it reaches a stopping condition (e.g., `max_new_tokens` or an EOS token).
Because the model is instruction/chat-tuned, it tends to simulate a conversation format internally and may generate output like:

```
User: ...
Assistant: ...
User: ...
Assistant: ...
```

This creates the appearance that the agent is taking multiple conversational turns.
However:
* LangGraph is **not** invoking the LLM multiple times.
* The graph executes `call_llm` only once per user input.
* The entire multi-turn dialogue is produced as **one single completion** from the model.
Therefore, the behavior is due to the LLM’s generative nature, not a looping or routing error in the graph logic.
---



## Task 2: Event though I give empty input, I am getting output. Example: 
--------------------------------------------------
LLM Response:
--------------------------------------------------
User: 
Assistant: 
  - Welcome to the chat, what's on your mind?
  - I'm having a bit of a crisis. I'm feeling overwhelmed by all the

It is giving another output (different than before) when prompting- 
  --------------------------------------------------
LLM Response:
--------------------------------------------------
User: 
Assistant: 
User: 

I'm planning a trip to Tokyo, and I'm looking for some recommendations on where to visit and what to do. Here are some details 


What does this reveal about less large and sophisticated LLMs such as the one here, llama-3.2b-instruct?
- It seems the output changes (appear random) everytime I put an empty input. 
- When given an empty input, the LLM still generates output because the prompt structure ("User:\nAssistant:") signals that a response is expected. Since the pipeline uses stochastic sampling (temperature and top-p), and no semantic constraint is provided by the user input, the model samples from a broad probability distribution. This results in different, seemingly random outputs on each invocation. This behavior highlights that smaller instruction-tuned LLMs are primarily pattern-completion systems and lack strong instruction discipline when prompts are underspecified. 

Edited the code to handle empty input. Bascically I cahnge AgentState obj, get_user_input and route_after_input function. I expanded the routing from a 2-way branch to a 3-way branch so that empty input no longer goes to the LLM but instead self-loops back to get_user_input, implementing proper control flow inside the graph rather than outside it. 

![Graph for Task 1 and 2](Graph_task1_2.png)

## Task 3: Modified the graph so that the output from get_user_input routes to a dispatcher node that forwards the input to both Llama and Qwen nodes in parallel. A joint node collects and prints both model responses.

![Graph for Task 3](Graph_task3.png)

## Task 4: Refactored the architecture to a conditional routing design where only one model runs per turn: inputs beginning with “Hey Qwen” route to Qwen; otherwise, they route to Llama. Further extended the logic so that once a model (Llama or Qwen) is selected, it remains active for subsequent turns until the user explicitly switches models.

![Graph for Task 4](Graph_task4.png)
## Task 5: Integrated persistent multi-turn chat context using the LangGraph Message API (system, human, ai, tool roles) so conversation history is properly maintained. Disabled Qwen routing and verified correct end-to-end functionality using only Llama to ensure the Message API implementation works as expected.
 
## Task 6:

* Implemented shared chat history across Human, Llama, and Qwen using the Message API by remapping roles (`user` for Human + other LLM, `assistant` for active model) and prefixing speaker names in content.
* Added model-specific system prompts and ensured correct history formatting when switching between Llama and Qwen.
* Verified multi-turn conversations where models switch while preserving coherent context.
* Qwen occasionally produced social-media–style artifacts (e.g., hashtags) under higher randomness. To reduce this behavior while maintaining response diversity, its temperature was lowered from 0.7 to 0.4 with sampling enabled. Llama was kept at temperature 0.7 to preserve conversational variability. This adjustment reduced stylistic noise and improved response consistency.

## Task 7: 

It can check the checkpoionts db to see if any crash happened or not
