Persona Chatbot - Local AI Chat Interface
A professional Flask-based web application that provides a chat interface powered by local GPT4All models. Customize AI personas and chat with your local LLM models privately.

https://img.shields.io/badge/AI-Chatbot-blue
https://img.shields.io/badge/Flask-2.3.3-green
https://img.shields.io/badge/GPT4All-Local%2520Models-orange

🌟 Features
🤖 Local AI Models: Run completely offline with GGUF model files

🎭 Persona Customization: Define custom AI personalities and behaviors

💬 Real-time Chat: Smooth, responsive chat interface

📱 Responsive Design: Works on desktop and mobile devices

🔒 Privacy-First: All data stays on your local machine

📊 Chat Management: Export conversations and clear history

⚡ Fast Responses: Optimized for Meta-Llama-3.1 and similar models

🚀 Quick Start
Prerequisites
Python 3.8+

8GB+ RAM (16GB recommended)

5GB+ free disk space for models

Installation
Clone or download the project

bash
git clone <repository-url>
cd persona-chat-mcp
Create virtual environment

bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
Download a model (if you don't have one)

bash
# Create models directory
mkdir models

# Download a recommended model (choose one)
wget https://gpt4all.io/models/gguf/mistral-7b-openorca.Q4_0.gguf -O models/mistral-7b-openorca.Q4_0.gguf
# OR
wget https://gpt4all.io/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf -O models/llama-3.1-8b.Q4_0.gguf
Running the Application
bash
python app.py
Open your browser to: http://localhost:5000

📁 Project Structure
text
persona-chat-mcp/
├── models/                 # GGUF model files
│   └── Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf
├── templates/              # HTML templates
│   └── index.html
├── app.py                  # Main Flask application
├── check_model.py          # Model diagnostic tool
├── main.py                 # Alternative launcher
├── requirements.txt        # Python dependencies
└── README.md              # This file
⚙️ Configuration
Environment Variables
bash
# Optional: Set custom model path
export GPT4ALL_MODEL_PATH="models/your-model.gguf"

# Optional: Adjust generation parameters
export GPT4ALL_MAX_TOKENS=1024
export GPT4ALL_TEMPERATURE=0.7
Supported Models
✅ Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf

✅ Mistral-7B-OpenOrca

✅ Hermes-2-Pro-Mistral

✅ Any GPT4All-compatible GGUF model

🎯 Usage
Setting Personas
In the left sidebar, modify the persona text

Click "Set Persona" to apply changes

The AI will now respond according to your specified personality

Example personas:

Career Coach: "You are an experienced career advisor specializing in tech industry transitions..."

Creative Writer: "You are a creative writing assistant with a flair for storytelling..."

Technical Expert: "You are a senior software engineer with 15 years of experience..."

Chat Features
Send Messages: Type in the input box and press Enter or click Send

Clear History: Remove all conversation history

Export Chat: Download conversation as text file

Real-time Status: Monitor model loading and connection status

🔧 Troubleshooting
Common Issues
Model won't load:

bash
# Check if model file exists
python check_model.py

# Reinstall GPT4All if needed
pip uninstall gpt4all -y
pip install gpt4all==1.0.12
Out of memory:

Close other applications

Use a smaller model (Q4_0 instead of Q8_0)

Reduce MAX_TOKENS in configuration

Slow responses:

Ensure you have adequate RAM

Use models quantized to lower precision (Q4_0)

Close background applications

Diagnostic Tools
Run the model diagnostic:

bash
python check_model.py
This will check:

Model file existence and size

GPT4All installation

Model loading capability

🛠️ Development
Adding New Features
The project uses a modular architecture:

app.py: Main Flask application with routes

Chat management handled by SimpleChatManager class

Model interaction through GPT4All wrapper

Frontend uses vanilla JavaScript with CSS Grid/Flexbox

Customizing the UI
Modify templates/index.html and the embedded CSS to customize:

Color scheme in :root CSS variables

Layout in .container grid templates

Chat bubbles in .message classes

Extending Functionality
Possible enhancements:

Add user authentication

Implement conversation persistence

Add model parameter controls (temperature, top-p)

Support for multiple model switching

File upload for document analysis

📊 Performance Tips
RAM: 8GB minimum, 16GB recommended for 7B models

Storage: SSD recommended for faster model loading

CPU: Modern multi-core processor

Model Size: Q4_0 quantization provides best performance/RAM balance

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is open source and available under the MIT License.

🙏 Acknowledgments
GPT4All for the local model interface

Meta Llama for the Llama models

Flask web framework

📞 Support
If you encounter any issues:

Check the troubleshooting section above

Run the diagnostic tool: python check_model.py

Ensure your model file is compatible and complete

Check that you have sufficient system resources

