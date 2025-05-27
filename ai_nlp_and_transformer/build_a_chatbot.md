Here’s a detailed step-by-step guide for **Project 4: Build a Chatbot**, covering rule-based, retrieval-based, and generative AI approaches:

---

### **Step 1: Set Up Environment**
1. **Install Tools**:
   ```bash
   # For OpenAI API
   pip install openai python-dotenv

   # For Hugging Face Transformers
   pip install transformers datasets torch

   # For Rasa (rule-based/retrieval chatbot)
   pip install rasa
   ```

2. **Get API Keys**:
   - [OpenAI API Key](https://platform.openai.com/api-keys)
   - [Hugging Face Token](https://huggingface.co/settings/tokens)

---

### **Step 2: Build a Simple GPT-3.5/4 Chatbot (OpenAI API)**
#### **Code**:
```python
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_with_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4"
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat_with_gpt(user_input)
    print(f"Bot: {response}")
```

#### **Key Concepts**:
- **Role-Based Prompting**: Use `system`, `user`, and `assistant` roles to guide the conversation.
- **Temperature**: Control creativity (`temperature=0` for deterministic, `0.7` for creative).

---

### **Step 3: Fine-Tune a Seq2Seq Model (Hugging Face)**
Use a pre-trained model like **Google’s T5** or **Meta’s BlenderBot** for custom dialogue.

#### **Code**:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained model (e.g., T5-small)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Example fine-tuning on custom dataset (e.g., Cornell Movie Dialogs)
from datasets import load_dataset
dataset = load_dataset("cornell_movie_dialogs")

# Tokenize and format data
def preprocess(example):
    input_text = "chat: " + example["utterance"]
    target_text = example["response"]
    model_inputs = tokenizer(input_text, truncation=True, max_length=128)
    labels = tokenizer(target_text, truncation=True, max_length=128).input_ids
    return {"input_ids": model_inputs.input_ids, "labels": labels}

dataset = dataset.map(preprocess, batched=True)

# Train
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)
trainer.train()

# Generate responses
def generate_response(text):
    inputs = tokenizer.encode("chat: " + text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_response("Hello!"))  # Output: "Hi! How can I help you?"
```

---

### **Step 4: Build a Rule-Based Chatbot with Rasa**
#### **1. Initialize Rasa Project**:
```bash
rasa init --no-prompt
```

#### **2. Define Intents & Responses**:
- Edit `data/nlu.yml`:
  ```yaml
  nlu:
  - intent: greet
    examples: |
      - Hi
      - Hello
      - Good morning

  - intent: goodbye
    examples: |
      - Bye
      - See you later
  ```

- Edit `data/responses.yml`:
  ```yaml
  responses:
    utter_greet:
      - text: "Hello! How can I help you?"
    utter_goodbye:
      - text: "Goodbye! Have a great day!"
  ```

#### **3. Train & Test**:
```bash
rasa train
rasa shell
```

---

### **Step 5: Hybrid Approach (Intent + Generative AI)**
Combine Rasa’s intent recognition with OpenAI’s API for dynamic responses.

#### **Code**:
```python
# rasa/actions/actions.py
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import openai

class ActionGenerateResponse(Action):
    def name(self):
        return "action_generate_response"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")
        
        # Use OpenAI for open-ended questions
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )
        dispatcher.utter_message(response.choices[0].message['content'])
        return []
```

#### **Update Domain**:
```yaml
# domain.yml
actions:
  - action_generate_response
```

---

### **Key Concepts Explained**
1. **Sequence-to-Sequence (Seq2Seq)**:
   - Encoder processes input text → Decoder generates output (e.g., T5, BART).
   - Used in chatbots for mapping user input to responses.

2. **Transfer Learning**:
   - Start with a pre-trained model (e.g., GPT-3.5, T5) and fine-tune on custom dialogue data.

3. **Intent Recognition**:
   - Classify user input into predefined categories (e.g., `greet`, `goodbye`).
   - Tools: Rasa, spaCy’s text classification.

4. **Dialogue Management**:
   - Track conversation state (e.g., slots, context) to guide responses.
   - Tools: Rasa’s stories and rules.

---

### **Evaluation & Improvement**
1. **Metrics**:
   - **BLEU Score**: For generative models (measures response similarity to ground truth).
   - **Intent Accuracy**: For Rasa (percentage of correctly classified intents).

2. **Enhancements**:
   - Add **sentiment analysis** to adjust response tone.
   - Integrate **knowledge bases** for factual responses (e.g., company FAQs).
   - Use **reinforcement learning** for dialogue policy optimization.

---

### **Resources**
- **OpenAI Documentation**: [Chat Completion API](https://platform.openai.com/docs/guides/text-generation)
- **Hugging Face Course**: [Fine-Tuning Models](https://huggingface.co/learn/nlp-course/chapter3/1)
- **Rasa Documentation**: [Dialogue Management](https://rasa.com/docs/rasa/)

By following these steps, you’ll build chatbots ranging from simple rule-based agents to sophisticated AI-driven assistants! 🤖💬
