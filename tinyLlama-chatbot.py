import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
)


def build_prompt(history, system_prompt):
    prompt = f"<|system|>\n{system_prompt}\n<|end|>\n"
    for turn in history:
        prompt += f"<|user|>\n{turn['user']}\n<|end|>\n"
        prompt += f"<|assistant|>\n{turn['assistant']}\n<|end|>\n"
    return prompt

# Initialize
system_prompt = "You are a helpful, multilingual AI assistant."
print("TinyLlama Chatbot is ready!")
print("Tip: You can chat in English, Hindi, French, Spanish, and more.")
print("Type 'export' to save your conversation, or 'quit' to exit.")

chat_history = []
assistant_greeting = "Hello! How can I help you today?"
print("Bot:", assistant_greeting)

for _ in range(20):  # limit to 20 exchanges 
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        print("Chat ended.")
        break
    if user_input.lower() == "export":
        
        with open("chat_history.txt", "w", encoding="utf-8") as f:
            f.write(f"System: {system_prompt}\n")
            for turn in chat_history:
                f.write(f"You: {turn['user']}\n")
                f.write(f"Bot: {turn['assistant']}\n")
        print("Chat history exported to 'chat_history.txt'.")
        continue

    
    prompt = build_prompt(chat_history, system_prompt)
    prompt += f"<|user|>\n{user_input}\n<|end|>\n<|assistant|>\n"

    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    print("Bot:", response)

    
    chat_history.append({"user": user_input, "assistant": response})

