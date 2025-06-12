# Spanish AI Tutor

This Python application provides an interactive, multimedia environment to practice your Spanish conversational skills with a local AI tutor. Powered by [Ollama](https://ollama.com/), it leverages advanced voice recognition, text-to-speech, and prompt customization to create a dynamic and supportive learning experience.

Local. No internet needed after first launch (Downloads most of the necessary packages automatically during first launch. Read further for full step-by-step setup instructions). 

## 🌟 Key Features

*   🗣️ **Real-time Conversation**: Chat with an AI tutor (`Maestro`) that responds in Spanish and provides instant English translations for every sentence, helping you learn in context.
*   🎙️ **Advanced Speech-to-Text**: Speak your responses and have them transcribed automatically using [Whisper](https://github.com/openai/whisper).
    *   **Intelligent Voice Detection**: Uses Silero VAD (Voice Activity Detection) to automatically detect when you start and stop speaking. No "push-to-talk" button needed!
    *   **Flexible**: Supports multiple languages and Whisper model sizes (from `tiny` to `large` and `turbo` variants for faster performance on CUDA).
*   🔊 **Text-to-Speech**: Listen to the AI's Spanish responses to practice your pronunciation and listening comprehension. You can select different system voices and adjust the speaking rate.
*   🚀 **Proactive Engagement**: If you pause for too long, the AI can gently re-engage you with a question, vocabulary suggestion, or encouragement. This "Auto-Continuation" feature is fully customizable.
*   🧩 **Customizable AI Personality**:
    *   **System Prompts**: Define the AI's core behavior, personality, and teaching style using a system prompt.
    *   **Prompt Management**: Save, load, and manage multiple system prompts and continuation prompts directly from the UI. Experiment with different learning scenarios!
*   🖼️ **Multimodal Chat**:
    *   **Image Support**: Attach images to your messages (via file dialog, drag & drop, or pasting from the clipboard) and ask the AI questions about them.
    *   **Document Support**: Load text from `.pdf`, `.txt`, and other text-based files directly into the input box to discuss specific content.
*   ⚙️ **User-Friendly Interface**: A clean and intuitive UI built with Tkinter, providing easy access to all features and settings.

## 📋 Requirements

Before you begin, ensure you have the following installed:

1.  **Python 3.8+**
2.  **Ollama**: The application requires a running Ollama instance.
    *   [Download and install Ollama](https://ollama.com/).
    *   Pull a model to use. The application is optimized for instruction-following models. `gemma3:27b` is a great choice.
        ```bash
        ollama pull gemma3:27b
        ```
3.  **Git**: To clone the repository.
4.  **(Linux Only) TTS Dependencies**: `pyttsx3` may require `espeak` and `ffmpeg`.
    ```bash
    # On Debian/Ubuntu
    sudo apt-get update && sudo apt-get install espeak ffmpeg
    ```
5.  **(Optional) NVIDIA GPU**: For significantly faster Whisper transcription, an NVIDIA GPU with the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed is recommended. The app will automatically fall back to CPU if a GPU is not detected.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt` file, you can install the packages directly:
    ```bash
    pip install ollama pillow pyttsx3 numpy torch torchaudio openai-whisper faster-whisper pyaudio PyMuPDF tkinterdnd2
    ```
    *Note: `torch` and `torchaudio` can be large downloads. If you have a CUDA-enabled GPU, follow the [official PyTorch instructions](https://pytorch.org/get-started/locally/) for the correct installation command to get GPU support.*

## ▶️ How to Run

1.  **Start Ollama**: Make sure the Ollama application is running in the background.

2.  **Run the Python script:**
    ```bash
    python SpanishTutorApp.py
    ```

3.  **First-Time Setup**: The first time you run the app and enable Voice Input, it will download the Silero VAD and Whisper models. This may take a few minutes depending on your internet connection.

## 🕹️ Using the Application

1.  **Select an Ollama Model**: Choose your desired model from the dropdown at the top-left.
2.  **Enable Features**:
    *   **Text-to-Speech**: Check "Enable TTS" to hear the AI's responses. Select a voice and adjust the speed.
    *   **Voice Input**: Check "Enable Voice" to start the VAD listener. The status indicator will show "Listening...". Simply start speaking, and it will record automatically.
3.  **Start a Conversation**:
    *   **Type**: Write a message in the input box and press `Enter` to send.
    *   **Speak**: If Voice Input is enabled, just speak. After you pause, your speech will be transcribed into the input box. If "Auto-send" is checked, it will be sent automatically.
4.  **Customize the AI**:
    *   **System Prompt**: Edit the text in the "System Prompt" box to change how the AI behaves. Click "Apply Selected" to make your changes active.
    *   **Save/Load Prompts**: Use the "Save As New Prompt" button to store your custom prompts for later use via the dropdown.
5.  **Attach Files**:
    *   **Images**: Drag an image file onto the preview area, use the "Open File" button, or simply paste an image from your clipboard (`Ctrl+V` or `Cmd+V`).
    *   **Documents**: Drag a `.pdf` or `.txt` file into the main text input area to load its content for discussion.

## 📂 Configuration Files

The application automatically creates two JSON files in the same directory to store your custom prompts:

*   `system_prompts.json`: Stores all your saved system prompts.
*   `continuation_prompts.json`: Stores all your saved prompts for the "Auto-Continuation" feature.

You can manually edit these files, but it's easier to manage them through the application's UI.

## 🔧 Troubleshooting

*   **Slow Transcription**: If transcription is slow, try using a smaller Whisper model size (e.g., `base` or `small`) or a `turbo` variant if you have a compatible GPU.
*   **TTS Not Working**:
    *   Ensure you have enabled it in the UI.
    *   On Linux, make sure `espeak` is installed.
    *   On Windows, some voices may not be compatible. Try selecting a different one (e.g., Microsoft David/Zira).
*   **VAD is Too Sensitive/Not Sensitive Enough**: The VAD is tuned for a quiet environment. Background noise can interfere with silence detection. Try to be in a quiet space for the best experience.
*   **`pyaudio` Installation Error**: `pyaudio` can sometimes be tricky to install. You may need to install `portaudio` on your system first.
    ```bash
    # On Debian/Ubuntu
    sudo apt-get install portaudio19-dev
    # On macOS (using Homebrew)
    brew install portaudio
    ```
    Then, try `pip install pyaudio` again.

## 📄 License

See the [LICENSE](LICENSE) file for details.
