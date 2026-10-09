### COMPLETE THE CODE ###

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from rag_tool import rag_tool

# ---- Load Qwen3-0.6B (local, from the SFT project) ----
MODEL_PATH = "../SCSE_26_GEX3/models/qwen3-0.6b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    device_map="auto",
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=30,
    do_sample=False,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
)
llm = HuggingFacePipeline(pipeline=pipe)

# ---- Use the RAG tool to retrieve policy context ----
question = "What exact phone number should I call for the University IT Service Desk?"

# Retrieve relevant policy chunks
context = rag_tool.func(question)

# Build the prompt with retrieved context
prompt = f"""You are a university IT support assistant.
Answer the user's question using ONLY the policy information below.
If the answer is not in the policy, say that the information is not specified.
Do NOT invent phone numbers, addresses, or any facts.

Policy information:
{context}

Question: {question}

Answer:"""

# Generate
result = llm.invoke(prompt)

# Extract only the answer (after "Answer:")
answer = result.split("Answer:")[-1].strip()

print("\n=== QUESTION ===")
print(question)
print("\n=== RETRIEVED CONTEXT ===")
print(context)
print("\n=== MODEL ANSWER ===")
print(answer)
