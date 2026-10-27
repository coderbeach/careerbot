
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# 1. Load the AI model
# -------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CareerCompass AI on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto"
).to(DEVICE)

model.eval()

print("CareerCompass AI is ready!")

# -------------------------------
# 2. Define the chatbot's role
# -------------------------------

SYSTEM_PROMPT = """
You are CareerCompass AI, a helpful and practical AI career
guidance assistant for students and beginners.

Your responsibilities:
- Recommend suitable career paths based on interests and skills.
- Explain career options such as AI Engineer, ML Engineer,
  Data Scientist and Software Developer.
- Create step-by-step learning roadmaps.
- Recommend beginner-friendly projects.
- Compare career paths and their required skills.
- Help students prepare for internships and placements.
- Ask follow-up questions when more information is needed.

Give clear, structured and actionable answers.
Explain technical concepts in beginner-friendly language.
Do not guarantee jobs, salaries or career outcomes.
Be honest when information is uncertain.
"""

# -------------------------------
# 3. Generate chatbot responses
# -------------------------------

def respond(message, history):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Include recent conversation for follow-up questions
    for item in history[-10:]:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")

            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({
                    "role": role,
                    "content": content
                })

    messages.append({
        "role": "user",
        "content": message
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return answer or "Sorry, I couldn't generate a response. Please try again."

# -------------------------------
# 4. Create the website interface
# -------------------------------

demo = gr.ChatInterface(
    fn=respond,
    title="CareerCompass AI 🚀",
    description=(
        "Your AI-powered career companion. "
        "Explore career paths, learning roadmaps, skills and projects."
    ),
    examples=[
        "What careers can I explore after CSE?",
        "Create a 3-month machine learning roadmap.",
        "Compare AI engineering and data science.",
        "Suggest three beginner AI projects.",
        "How can I prepare for an AI internship?"
    ],
    textbox=gr.Textbox(
        placeholder="Ask me about your career...",
        container=False
    )
)

# -------------------------------
# 5. Launch the application
# -------------------------------

if __name__ == "__main__":
    demo.launch()
