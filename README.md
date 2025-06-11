Of course! Here is a comprehensive and well-structured GitHub description for your SpanishTutorApp.py. This description is written in Markdown format, perfect for a README.md file.

Spanish Tutor AI 🇪🇸🤖

A desktop application for practicing Spanish conversation with a local AI, featuring real-time voice recognition (Speech-to-Text) and spoken responses (Text-to-Speech). Built with Python, Ollama, and Whisper.

![image](https://github.com/user-attachments/assets/9d33302f-6c15-46cf-8cce-8425f925488f)


✨ Core Features

This application is designed to be a fully-featured, all-in-one tool for language practice.

💬 Interactive Conversation: Chat with a local AI model running via Ollama. The AI is guided by a customizable system prompt to act as a patient and friendly Spanish tutor.

🗣️ Voice-to-Text (Speech Recognition): Speak directly to the application! It uses advanced Voice Activity Detection (VAD) to listen for when you're talking and automatically transcribes your speech using Whisper.

Supports both the standard whisper and the highly optimized faster-whisper libraries.

Choose from various Whisper model sizes (tiny to large, including turbo models) to balance speed and accuracy.

🔊 Text-to-Speech (TTS): The AI's responses are spoken aloud, allowing you to practice your listening comprehension.

Easily toggle TTS on or off.

Select from any of your system's installed voices.

Adjust the AI's talking speed to your comfort level.

** bilingual Responses:** The default AI prompt is configured to provide every Spanish sentence with an immediate English translation, creating a powerful learning loop.

AI Response Example:
¡Hola! Estoy muy bien, gracias.
[EN] Hello! I'm very well, thank you.
¿Y tú? ¿Qué tal tu día?
[EN] And you? How is your day?

🖼️ Multimodal Input: Interact with the AI using more than just text.

Attach Images: Send images to the AI and ask questions about them.

Load Files: Open and load the content from .pdf, .txt, .md, and .py files directly into the input box.

Drag & Drop: Simply drag files onto the application window to load them.

Paste from Clipboard: Paste images directly from your clipboard with Ctrl+V.

⚙️ Highly Customizable:

Ollama Model Selection: Dynamically fetches and allows you to switch between any of your installed Ollama models.

System Prompt Editor: Modify the AI's core behavior, or create, save, and load your own custom system prompts for different learning scenarios.

🚀 Responsive UI: Built with a multi-threaded architecture to ensure the user interface remains smooth and responsive, even while models are loading or the AI is "thinking".

🛠️ Technology Stack

AI Backend: Ollama

GUI: Tkinter (with tkinterdnd2 for drag-and-drop)

Speech-to-Text: OpenAI Whisper (via openai-whisper and faster-whisper libraries)

Voice Activity Detection (VAD): Silero VAD

Text-to-Speech (TTS): pyttsx3

Audio I/O: PyAudio

File Handling: Pillow (for images), PyMuPDF (for PDFs)

🏁 Getting Started
Prerequisites

Python 3.8+

Ollama Installed and Running: You must have the Ollama application installed and running on your machine.

An Ollama Model: Pull a model that the application can use. Multimodal models are recommended for image support.

# We recommend a versatile model like gemma3 or llama3
ollama pull gemma3:27b


FFmpeg (for Whisper): The Whisper library requires FFmpeg. Follow the official instructions for your OS: https://ffmpeg.org/download.html. On Windows, you can use winget install ffmpeg. On macOS, brew install ffmpeg. On Linux, sudo apt install ffmpeg.

PortAudio (for PyAudio): PyAudio may require PortAudio development libraries.

On Debian/Ubuntu: sudo apt-get install portaudio19-dev

On macOS (with Homebrew): brew install portaudio

Installation

Clone the repository:

git clone https://github.com/your-username/SpanishTutorApp.git
cd SpanishTutorApp
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Create and activate a virtual environment (recommended):

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install the required Python packages:

pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

If a requirements.txt is not provided, install the packages manually:

pip install ollama tkinterdnd2 Pillow pyttsx3 openai-whisper faster-whisper pyaudio torch torchaudio PyMuPDF
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Note: torch and torchaudio are large libraries required for Whisper and VAD).

Running the Application

With your virtual environment activated and Ollama running in the background, simply run the script:

python SpanishTutorApp.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
kullanım

Select an AI Model: Choose an available Ollama model from the top-left dropdown.

Enable Voice Features: The "Enable TTS" and "Enable Voice" checkboxes are on by default. You can toggle them as needed. The status indicator below the voice controls will show if it's listening, recording, or processing.

Customize the AI (Optional): Edit the system prompt to change the AI's personality or instructions. You can save your favorite prompts using the "Save" button.

Start the Conversation:

Type: Write your message in the bottom input box and press Enter or click "Send".

Speak: If voice input is enabled, just start talking! The app will detect your speech, transcribe it, and (by default) automatically send it to the AI.

Attach a File: Click "Open File" or drag a supported file onto the application to discuss its contents.

Enjoy practicing your Spanish
