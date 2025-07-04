## TinyLlama Chatbot

A lightweight conversational AI chatbot powered by TinyLlama-1.1B-Chat-v1.0 model from Hugging Face Transformers. This chatbot supports multilingual conversations and maintains chat history throughout the session.

## Features

- Multilingual support (English, Hindi, French, Spanish, and more)
- Conversation history maintenance
- Chat export functionality
- Role-based prompt formatting
- Configurable response generation parameters
- Memory-efficient implementation

## System Requirements

### Minimal Requirements

- Python 3.8 or higher
- 4GB RAM (minimum)
- 8GB available disk space
- Internet connection for initial model download

### Recommended Requirements

- Python 3.9 or higher
- 8GB RAM or more
- 16GB available disk space
- CUDA-compatible GPU (optional, for faster inference)

## Installation

### Step 1: Clone or Download

Save the provided Python script as `tinyllama_chatbot.py` in your desired directory.

### Step 2: Install Dependencies

#### Option A: Using pip (Recommended)

```bash
pip install torch transformers
```

#### Option B: Using conda

```bash
conda install pytorch transformers -c pytorch -c huggingface
```

#### Option C: For Google Colab

```bash
!pip install torch transformers --quiet
```

### Step 3: Verify Installation

Run the following command to verify your installation:

```python
import torch
import transformers
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
```

## Usage

### Basic Usage

1. Navigate to the directory containing `tinyllama_chatbot.py`
2. Run the script:

```bash
python tinyllama_chatbot.py
```

3. Wait for the model to download and load (first run only)
4. Start chatting when you see "TinyLlama Chatbot is ready!"

### Available Commands

- **Normal chat**: Simply type your message and press Enter
- **Export chat**: Type `export` to save conversation history to `chat_history.txt`
- **Exit**: Type `quit` to end the chat session

### Example Session

```
TinyLlama Chatbot is ready!
Tip: You can chat in English, Hindi, French, Spanish, and more.
Type 'export' to save your conversation, or 'quit' to exit.
Bot: Hello! How can I help you today?
You: What is the capital of France?
Bot: The capital of France is Paris.
You: Can you tell me a joke?
Bot: Sure! Why don't scientists trust atoms? Because they make up everything!
You: export
Chat history exported to 'chat_history.txt'.
You: quit
Chat ended.
```

### Output Example

![alt text](<WhatsApp Image 2025-07-04 at 23.20.34_9d2f942b.jpg>)
![alt text](<WhatsApp Image 2025-07-04 at 23.20.34_644e5edc.jpg>)
![alt text](<WhatsApp Image 2025-07-04 at 23.20.34_89ec5f30.jpg>)

## Configuration

### Customizing System Prompt

Modify the `system_prompt` variable to change the chatbot's behavior:

```python
system_prompt = "You are a helpful coding assistant specialized in Python."
```

### Adjusting Response Parameters

Modify the generation parameters in the `model.generate()` call:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,        # Increase for longer responses
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.9,           # Higher = more creative, Lower = more focused
    top_p=0.95,               # Nucleus sampling parameter
    repetition_penalty=1.1     # Reduce repetition
)
```

### Extending Conversation Length

Change the loop limit for longer conversations:

```python
for _ in range(50):  # Allow 50 exchanges instead of 20
```

## Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'transformers'"

**Solution**: Install transformers library

```bash
pip install transformers
```

#### Issue: "RuntimeError: No such file or directory"

**Solution**: Ensure stable internet connection for model download. The model (approximately 2.2GB) will be downloaded on first run.

#### Issue: "CUDA out of memory"

**Solution**: Add CPU-only execution:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu"  # Force CPU usage
)
```

#### Issue: Slow response generation

**Solutions**:

1. Reduce `max_new_tokens` parameter
2. Use GPU if available
3. Reduce conversation history length

#### Issue: Repetitive or poor quality responses

**Solutions**:

1. Adjust temperature (0.1-1.0 range)
2. Add repetition penalty
3. Modify system prompt for better guidance

### Memory Management

If experiencing memory issues:

1. **Reduce max_new_tokens**: Lower the token limit for responses
2. **Clear chat history**: Restart the program periodically
3. **Use smaller batch sizes**: Modify the generation parameters

### Performance Optimization

#### For CPU-only systems:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)
```

#### For GPU systems:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

## Advanced Usage

### Custom Model Loading

Replace the model name to use different models:

```python
model_name = "microsoft/DialoGPT-medium"  # Alternative model
```

### Batch Processing

For processing multiple inputs:

```python
def process_batch(inputs_list):
    responses = []
    for user_input in inputs_list:
        # Generate response for each input
        # ... (generation code)
        responses.append(response)
    return responses
```

### Integration with Web Framework

Example Flask integration:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    # ... (generation code)
    return jsonify({'response': response})
```

## File Structure

```
project-directory/
├── tinyllama_chatbot.py    # Main chatbot script
├── chat_history.txt        # Exported chat history (generated)
├── README.md              # This file
└── requirements.txt       # Dependencies (optional)
```

## Dependencies

Create a `requirements.txt` file:

```
torch>=1.9.0
transformers>=4.21.0
```

Install with:

```bash
pip install -r requirements.txt
```

## Model Information

- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Size**: ~2.2GB
- **Parameters**: 1.1 billion
- **Languages**: Multilingual support
- **License**: Apache 2.0

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:

1. Check the troubleshooting section
2. Verify your Python and dependency versions
3. Check available system memory
4. Review the Hugging Face Transformers documentation

## Acknowledgments

- Hugging Face for the Transformers library
- TinyLlama team for the pre-trained model
- PyTorch team for the underlying framework
