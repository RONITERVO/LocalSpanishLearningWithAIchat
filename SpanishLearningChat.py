# SpanishTutorApp.py
import ollama
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from PIL import Image, ImageTk
import pyttsx3
import threading
import queue
import re
import time
import whisper
import faster_whisper
import pyaudio
import wave
import numpy as np
import tempfile
import os
import torch
import torchaudio
from collections import deque
import traceback
import fitz
from tkinterdnd2 import TkinterDnD, DND_FILES
from typing import Optional
import json


# ===================
# Constants
# ===================
# --- Audio ---
CHUNK = 1024            # Audio frames per buffer
VAD_CHUNK = 512         # Smaller chunk for VAD responsiveness
RATE = 16000            # Sampling rate (must be 16 kHz for Silero VAD & good for Whisper)
FORMAT = pyaudio.paInt16
CHANNELS = 1
SILENCE_THRESHOLD_SECONDS = 1.0 # How long silence triggers end of recording
MIN_SPEECH_DURATION_SECONDS = 0.2 # Ignore very short bursts
PRE_SPEECH_BUFFER_SECONDS = 0.3 # Keep audio before speech starts


# --- UI ---
APP_TITLE = "Spanish Conversation Practice with AI"

# --- Models ---
DEFAULT_OLLAMA_MODEL = "No model selected" 
DEFAULT_WHISPER_MODEL_SIZE = "turbo-large" # Faster startup, change to 'medium' or 'large' for better accuracy

# --- Conversation Continuation ---
USER_INACTIVITY_CHECK_INTERVAL = 20000   # Check every 20 seconds
USER_INACTIVITY_THRESHOLD = 60000       # Prompt after 60 seconds of inactivity (in milliseconds)

# --- Whisper ---
WHISPER_LANGUAGES = [
    ("Auto Detect", None), ("English", "en"), ("Spanish", "es"), ("Finnish", "fi"), ("Swedish", "sv"),
    ("German", "de"), ("French", "fr"), ("Italian", "it"),
    ("Russian", "ru"), ("Chinese", "zh"), ("Japanese", "ja")
]
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "turbo-tiny", "turbo-base", "turbo-small", "turbo-medium", "turbo-large"]

CONTINUATION_PROMPTS_FILE = "continuation_prompts.json"
# Default continuation prompt when user is inactive
DEFAULT_CONTINUATION_PROMPT = """When continuing the conversation, adopt a supportive teacher mindset. The student may be thinking of vocabulary or formulating their thoughts in Spanish. Proactively help them by:

1. Asking a simple follow-up question related to the previous topic
2. Offering vocabulary suggestions they might need to respond
3. Providing sentence starters or examples they could modify
4. Introducing a related but slightly simpler concept if they seem stuck
5. Using encouraging phrases like "Puedes hacerlo" or "Toma tu tiempo"
6. Breaking down complex ideas into smaller, manageable questions

Be patient, encouraging, and give them tools to succeed. Remember that language learning requires thinking time, and silence may indicate the student is processing.

IMPORTANT: Do not mention or acknowledge any pause in the conversation. Just continue naturally with helpful guidance.

Remember to follow your primary rules: reply in Spanish with English translations."""

SYSTEM_PROMPTS_FILE = "system_prompts.json"
# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a friendly and patient Spanish language tutor AI. Your name is 'Maestro'.
Your goal is to have a natural, encouraging conversation with the user to help them practice Spanish. The user may speak in English, Spanish, or a mix of both.

**Your rules are:**
1.  **Always Reply in Spanish:** Your final output to the user must be in Spanish.
2.  **Keep it Concise:** Keep your responses concise and suitable for a beginner to intermediate learner.
3.  **Provide Translations:** For **every single Spanish sentence** you write, you MUST provide an English translation on the very next line, prefixed with `[EN]`.
4.  **Use Newlines:** Each Spanish sentence and each English translation must be on its own line. This is very important for the application to work correctly.

**Example 1: User asks for a translation.**
User: "How do I say 'I want to learn'?"
Your Response:
Se dice "Quiero aprender".
[EN] You say "I want to learn".
¿Qué más te gustaría saber?
[EN] What else would you like to know?

**Example 2: Conversational turn.**
User: "Hola, ¿cómo estás?"
Your Response:
¡Hola! Estoy muy bien, gracias.
[EN] Hello! I'm very well, thank you.
¿Y tú? ¿Qué tal tu día?
[EN] And you? How is your day?
"""

# ===================
# Globals
# ===================
# --- Ollama & Chat ---
messages = []
selected_image_path = ""
image_sent_in_history = False
current_model = DEFAULT_OLLAMA_MODEL
stream_queue = queue.Queue()
stream_done_event = threading.Event()
stream_in_progress = False
chat_history_widget = None
system_prompt_input_widget = None
system_prompt_selector = None
saved_system_prompts = {}
thinking_start_index = None
stream_buffer = ""
is_in_think_block = False

# --- Inactivity Settings ---
inactivity_enabled = None  # BooleanVar for enabling/disabling inactivity prompts
inactivity_threshold_seconds = None  # IntVar for threshold in seconds
continuation_prompt = None  # StringVar for the continuation prompt text
saved_continuation_prompts = {}
continuation_prompt_selector = None
prompt_text_widget = None

# --- Inactivity Tracking ---
last_user_activity_time = 0
inactivity_timer_id = None
inactivity_prompt_sent = False

# --- TTS ---
tts_engine = None
tts_queue = queue.Queue()
tts_thread = None
tts_enabled = None
tts_rate = None
tts_voice_id = None
tts_busy = False
tts_busy_lock = threading.Lock()
tts_initialized_successfully = False

# --- Whisper/VAD ---
vad_model = None
vad_utils = None
vad_get_speech_ts = None
whisper_model = None
whisper_model_size = DEFAULT_WHISPER_MODEL_SIZE
whisper_initialized = False
vad_initialized = False
vad_thread = None
vad_stop_event = threading.Event()
whisper_queue = queue.Queue()
whisper_processing_thread = None
whisper_language = None
voice_enabled = None
recording_indicator_widget = None
auto_send_after_transcription = None
user_input_widget = None

# --- Audio Handling ---
py_audio = None
audio_stream = None
is_recording_for_whisper = False
audio_frames_buffer = deque()
vad_audio_buffer = deque(maxlen=int(RATE / VAD_CHUNK * 1.5))
temp_audio_file_path = None

# ===========================
# Conversation Continuation
# ===========================

def load_continuation_prompts():
    """Loads continuation prompts from the JSON file."""
    global saved_continuation_prompts
    try:
        if os.path.exists(CONTINUATION_PROMPTS_FILE):
            with open(CONTINUATION_PROMPTS_FILE, 'r') as f:
                saved_continuation_prompts = json.load(f)
            print(f"[Prompts] Loaded {len(saved_continuation_prompts)} continuation prompts.")
    except (json.JSONDecodeError, IOError) as e:
        print(f"[Prompts] Error loading continuation prompts file: {e}")
        saved_continuation_prompts = {}

def save_continuation_prompts_to_file():
    """Saves the current continuation prompts to the JSON file."""
    global saved_continuation_prompts
    try:
        with open(CONTINUATION_PROMPTS_FILE, 'w') as f:
            json.dump(saved_continuation_prompts, f, indent=4)
        print("[Prompts] Continuation prompts saved to file.")
    except IOError as e:
        print(f"[Prompts] Error saving continuation prompts file: {e}")

def populate_continuation_prompt_selector():
    """Updates the continuation prompt dropdown with loaded prompts."""
    if continuation_prompt_selector:
        prompt_names = list(saved_continuation_prompts.keys())
        continuation_prompt_selector['values'] = [""] + prompt_names
        continuation_prompt_selector.set("")

def on_continuation_prompt_selected(event=None):
    """Handles selection from the continuation prompt dropdown."""
    global prompt_text_widget
    selected_name = continuation_prompt_selector.get()
    if selected_name and selected_name in saved_continuation_prompts:
        prompt_text = saved_continuation_prompts[selected_name]
        continuation_prompt.set(prompt_text)
        prompt_text_widget.delete("1.0", tk.END)
        prompt_text_widget.insert("1.0", prompt_text)

def save_current_continuation_prompt():
    """Saves the text in the input box as a new named continuation prompt."""
    from tkinter.simpledialog import askstring
    global prompt_text_widget, saved_continuation_prompts
    
    prompt_text = prompt_text_widget.get("1.0", tk.END).strip()
    if not prompt_text:
        add_message_to_ui("error", "Cannot save an empty continuation prompt.")
        return

    prompt_name = askstring("Save Prompt", "Enter a name for this continuation prompt:")
    if prompt_name:
        saved_continuation_prompts[prompt_name] = prompt_text
        save_continuation_prompts_to_file()
        populate_continuation_prompt_selector()
        continuation_prompt_selector.set(prompt_name)
        add_message_to_ui("status", f"Continuation prompt '{prompt_name}' saved.")

def delete_selected_continuation_prompt():
    """Deletes the currently selected continuation prompt."""
    from tkinter.messagebox import askyesno
    global continuation_prompt_selector, prompt_text_widget
    
    selected_name = continuation_prompt_selector.get()
    if not selected_name:
        add_message_to_ui("error", "No continuation prompt selected to delete.")
        return

    if askyesno("Confirm Delete", f"Are you sure you want to delete the prompt '{selected_name}'?"):
        if selected_name in saved_continuation_prompts:
            del saved_continuation_prompts[selected_name]
            save_continuation_prompts_to_file()
            populate_continuation_prompt_selector()
            add_message_to_ui("status", f"Continuation prompt '{selected_name}' deleted.")

def reset_user_activity_timer():
    """Resets the timer tracking user's last activity."""
    global last_user_activity_time, inactivity_prompt_sent
    last_user_activity_time = time.time() * 1000
    inactivity_prompt_sent = False

def check_user_inactivity():
    """Checks if the user has been inactive and prompts the AI if needed."""
    global last_user_activity_time, inactivity_timer_id, inactivity_prompt_sent, stream_in_progress, messages, tts_busy, tts_busy_lock

    # Skip check if inactivity prompts are disabled
    if not inactivity_enabled.get():
        inactivity_timer_id = window.after(USER_INACTIVITY_CHECK_INTERVAL, check_user_inactivity)
        return

    current_time = time.time() * 1000
    elapsed_time = current_time - last_user_activity_time
    
    # Convert threshold from seconds to milliseconds for comparison
    threshold_ms = inactivity_threshold_seconds.get() * 1000
    
    with tts_busy_lock:
        is_tts_active = tts_busy
    
    # Conditions to prompt:
    # 1. Elapsed time is over the threshold.
    # 2. We haven't already sent an inactivity prompt for this silence period.
    # 3. The AI is not currently generating a response.
    # 4. There is an existing conversation to continue (messages list is not empty).
    # 5. The AI is not currently speaking.
    if elapsed_time > threshold_ms and not inactivity_prompt_sent and not stream_in_progress and messages and not is_tts_active:
        print(f"[Inactivity] User has been inactive for {elapsed_time/1000:.1f}s, prompting AI to continue.")
        inactivity_prompt_sent = True # Mark that we've sent the prompt
        prompt_ai_for_continuation()

    # Schedule the next check
    inactivity_timer_id = window.after(USER_INACTIVITY_CHECK_INTERVAL, check_user_inactivity)

def prompt_ai_for_continuation():
    """Sends a special system message to the AI to continue the conversation."""
    global stream_in_progress, stream_done_event, thinking_start_index

    if stream_in_progress:
        return # Don't interrupt if the AI is already talking

    # Use the custom continuation prompt
    continuation_prompt_text = continuation_prompt.get()
    
    # Get the main system prompt from the UI
    system_prompt_text = system_prompt_input_widget.get("1.0", tk.END).strip()
    
    # Combine the main system prompt with our special continuation instruction for this one call
    # This special instruction is NOT saved to the main history.
    temp_system_content = f"{system_prompt_text}\n\n[Internal Instruction]: {continuation_prompt_text}"

    # Set up the stream and "Thinking..." indicator
    send_button.config(state=tk.DISABLED)
    stream_in_progress = True
    stream_done_event.clear()

    chat_history_widget.config(state=tk.NORMAL)
    thinking_start_index = chat_history_widget.index(tk.INSERT)
    chat_history_widget.insert(tk.INSERT, "Thinking...\n", "thinking")
    chat_history_widget.see(tk.END)
    chat_history_widget.config(state=tk.DISABLED)
    window.update_idletasks()

    # Stop any ongoing TTS
    if tts_engine and tts_enabled.get():
        try: tts_engine.stop()
        except: pass
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break
        with tts_busy_lock: tts_busy = False

    # Start the chat worker thread. Notice we pass an empty string for the user message
    # and our special combined prompt for the system prompt. The user message is left empty in the history.
    thread = threading.Thread(
        target=chat_worker, 
        args=("", temp_system_content, None), 
        daemon=True
    )
    thread.start()

    # This part is the same as in send_message, to re-enable the button when done
    def check_done():
        if stream_done_event.is_set():
            send_button.config(state=tk.NORMAL)
        else:
            window.after(100, check_done)

    window.after(50, process_stream_queue)
    window.after(100, check_done)

def create_inactivity_settings_ui():
    """Create UI controls for inactivity settings"""
    global inactivity_enabled, inactivity_threshold_seconds, continuation_prompt
    global continuation_prompt_selector, prompt_text_widget

    # Main collapsible frame
    inactivity_header_button = ttk.Button(
        controls_sidebar_frame,
        text="Auto-Continuation ▸",
        command=lambda: toggle_frame(inactivity_frame, inactivity_header_button)
    )
    inactivity_header_button.pack(fill=tk.X, pady=(0, 5))
    inactivity_frame = tk.Frame(controls_sidebar_frame, bg="#F5F5F7")
    
    # --- Enable Checkbox ---
    inactivity_checkbox = ttk.Checkbutton(
        inactivity_frame, 
        text="Continue after user inactivity", 
        variable=inactivity_enabled,
        command=lambda: threshold_scale.configure(state=tk.NORMAL if inactivity_enabled.get() else tk.DISABLED)
    )
    inactivity_checkbox.pack(anchor="w", pady=(0, 10))

    # --- Threshold Slider ---
    threshold_value = tk.StringVar(value=f"{inactivity_threshold_seconds.get()} seconds")
    threshold_title_frame = tk.Frame(inactivity_frame, bg="#F5F5F7")
    threshold_title_frame.pack(fill=tk.X)
    tk.Label(threshold_title_frame, text="Wait Time:", bg="#F5F5F7", font=("Arial", 8)).pack(side=tk.LEFT)
    tk.Label(threshold_title_frame, textvariable=threshold_value, bg="#F5F5F7", font=("Arial", 8)).pack(side=tk.RIGHT)

    threshold_scale = ttk.Scale(
        inactivity_frame,
        from_=10, to=300, orient=tk.HORIZONTAL, variable=inactivity_threshold_seconds,
        command=lambda v: threshold_value.set(f"{int(float(v))} seconds")
    )
    threshold_scale.pack(fill=tk.X, pady=(2, 10))
    threshold_scale.configure(state=tk.NORMAL if inactivity_enabled.get() else tk.DISABLED)

    # --- Saved Prompts ---
    tk.Label(inactivity_frame, text="Continuation Instruction Preset:", bg="#F5F5F7", font=("Arial", 8), anchor="w").pack(fill=tk.X)
    continuation_prompt_selector = ttk.Combobox(inactivity_frame, state="readonly", font=("Arial", 8))
    continuation_prompt_selector.pack(fill=tk.X, pady=2)
    continuation_prompt_selector.bind("<<ComboboxSelected>>", on_continuation_prompt_selected)
    
    # --- Prompt Text Area ---
    tk.Label(inactivity_frame, text="AI Instructions:", anchor="w", bg="#F5F5F7", font=("Arial", 8)).pack(fill=tk.X, pady=(10, 0))
    prompt_text_widget = scrolledtext.ScrolledText(inactivity_frame, wrap=tk.WORD, height=5, font=("Arial", 9))
    prompt_text_widget.pack(fill=tk.X, pady=2)
    prompt_text_widget.insert("1.0", continuation_prompt.get())
    
    def update_prompt_var(event=None):
        continuation_prompt.set(prompt_text_widget.get("1.0", tk.END).strip())
    prompt_text_widget.bind("<KeyRelease>", update_prompt_var)
    
    # --- Action Buttons ---
    buttons_frame = tk.Frame(inactivity_frame, bg="#F5F5F7")
    buttons_frame.pack(fill=tk.X, pady=(5,0))

    save_button = tk.Button(
        buttons_frame, text="Save Current Text as New...", command=save_current_continuation_prompt,
        font=("Arial", 8, "bold"), relief=tk.FLAT, bg="#d0e0ff"
    )
    save_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
    
    delete_button = tk.Button(
        buttons_frame, text="Delete Selected Preset", command=delete_selected_continuation_prompt,
        font=("Arial", 8), relief=tk.FLAT
    )
    delete_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

    return inactivity_frame

#============================
#System Prompt Management
#============================

def load_system_prompts():
    """Loads system prompts from the JSON file."""
    global saved_system_prompts
    try:
        if os.path.exists(SYSTEM_PROMPTS_FILE):
            with open(SYSTEM_PROMPTS_FILE, 'r') as f:
                saved_system_prompts = json.load(f)
            print(f"[Prompts] Loaded {len(saved_system_prompts)} system prompts.")
    except (json.JSONDecodeError, IOError) as e:
        print(f"[Prompts] Error loading system prompts file: {e}")
        saved_system_prompts = {}

def save_system_prompts_to_file():
    """Saves the current system prompts to the JSON file."""
    global saved_system_prompts
    try:
        with open(SYSTEM_PROMPTS_FILE, 'w') as f:
            json.dump(saved_system_prompts, f, indent=4)
        print("[Prompts] System prompts saved to file.")
    except IOError as e:
        print(f"[Prompts] Error saving system prompts file: {e}")

def populate_system_prompt_selector():
    """Updates the system prompt dropdown with loaded prompts."""
    if system_prompt_selector:
        prompt_names = list(saved_system_prompts.keys())
        system_prompt_selector['values'] = [""] + prompt_names
        system_prompt_selector.set("")

def on_system_prompt_selected(event=None):
    """Handles selection from the system prompt dropdown."""
    selected_name = system_prompt_selector.get()
    if system_prompt_input_widget:
        system_prompt_input_widget.delete("1.0", tk.END)
        if selected_name and selected_name in saved_system_prompts:
            prompt_text = saved_system_prompts[selected_name]
            system_prompt_input_widget.insert("1.0", prompt_text)

def save_current_system_prompt():
    """Saves the text in the input box as a new named prompt."""
    from tkinter.simpledialog import askstring
    prompt_text = system_prompt_input_widget.get("1.0", tk.END).strip()
    if not prompt_text:
        add_message_to_ui("error", "Cannot save an empty system prompt.")
        return

    prompt_name = askstring("Save Prompt", "Enter a name for this system prompt:")
    if prompt_name:
        saved_system_prompts[prompt_name] = prompt_text
        save_system_prompts_to_file()
        populate_system_prompt_selector()
        system_prompt_selector.set(prompt_name)
        add_message_to_ui("status", f"System prompt '{prompt_name}' saved.")

def delete_selected_system_prompt():
    """Deletes the currently selected system prompt."""
    from tkinter.messagebox import askyesno
    selected_name = system_prompt_selector.get()
    if not selected_name:
        add_message_to_ui("error", "No system prompt selected to delete.")
        return

    if askyesno("Confirm Delete", f"Are you sure you want to delete the prompt '{selected_name}'?"):
        if selected_name in saved_system_prompts:
            del saved_system_prompts[selected_name]
            save_system_prompts_to_file()
            populate_system_prompt_selector()
            system_prompt_input_widget.delete("1.0", tk.END)
            add_message_to_ui("status", f"System prompt '{selected_name}' deleted.")

def clear_system_prompt_fields():
    """Clears the system prompt input and selection."""
    system_prompt_selector.set("")
    system_prompt_input_widget.delete("1.0", tk.END)
    system_prompt_input_widget.insert("1.0", DEFAULT_SYSTEM_PROMPT)


def apply_system_prompt():
    """
    Handles the 'Apply' button click for system prompts.
    It prioritizes activating the prompt selected in the dropdown.
    - If a valid prompt is selected in the dropdown, its text is loaded/reloaded into the input widget and confirmed as active.
    - If no specific prompt is selected in the dropdown (e.g., blank option) BUT the text box has content, that custom text is confirmed as active.
    - If both the dropdown selection is blank/invalid and the text box is empty, the DEFAULT_SYSTEM_PROMPT is loaded and confirmed as active.
    The system prompt actually used by the AI is always what's in system_prompt_input_widget when a message is sent.
    """
    global system_prompt_selector, system_prompt_input_widget, saved_system_prompts, DEFAULT_SYSTEM_PROMPT
    
    selected_name_in_dropdown = system_prompt_selector.get()
    current_text_in_input_widget = system_prompt_input_widget.get("1.0", tk.END).strip()

    if selected_name_in_dropdown and selected_name_in_dropdown in saved_system_prompts:
        # A specific, named prompt is selected in the dropdown. This takes precedence.
        prompt_text_from_dropdown = saved_system_prompts[selected_name_in_dropdown]
        if current_text_in_input_widget != prompt_text_from_dropdown:
            system_prompt_input_widget.delete("1.0", tk.END)
            system_prompt_input_widget.insert("1.0", prompt_text_from_dropdown)
            add_message_to_ui("status", f"System prompt '{selected_name_in_dropdown}' from dropdown has been reloaded and is now active.")
            prompt_status.config(text=f"Active: '{selected_name_in_dropdown}'", fg="green")
        else:
            add_message_to_ui("status", f"System prompt '{selected_name_in_dropdown}' from dropdown is active.")
            prompt_status.config(text=f"Active: '{selected_name_in_dropdown}'", fg="green")
    elif current_text_in_input_widget:
        # No valid/specific prompt selected in dropdown (it's "" or invalid), but the text box has content.
        add_message_to_ui("status", "Custom system prompt from text input is active.")
        prompt_status.config(text="Active: Custom unsaved prompt", fg="green")
    else:
        # No specific prompt in dropdown AND text input is empty.
        # This typically happens if the user selected the blank option in the dropdown (which clears the text box).
        # In this case, load and activate the DEFAULT_SYSTEM_PROMPT.
        system_prompt_input_widget.delete("1.0", tk.END)
        system_prompt_input_widget.insert("1.0", DEFAULT_SYSTEM_PROMPT)
        add_message_to_ui("status", "Default system prompt has been loaded and is now active.")
        prompt_status.config(text="Active: Default system prompt", fg="green")



def on_text_change(event=None):
    """Indicate when the text has been modified but not applied"""
    current_selection = system_prompt_selector.get()
    if current_selection:
        original = saved_system_prompts.get(current_selection, "")
        current = system_prompt_input_widget.get("1.0", "end-1c")
        if original != current:
            prompt_status.config(text=f"Modified: '{current_selection}' (not applied)", fg="orange")
        else:
            prompt_status.config(text=f"Active: '{current_selection}'", fg="green")
    else:
        prompt_status.config(text="Custom prompt (not applied)", fg="orange")

# ===================
# TTS Setup & Control
# ===================
def initialize_tts():
    """Initializes the TTS engine. Returns True on success."""
    global tts_engine, tts_initialized_successfully, tts_rate, tts_voice_id
    if tts_engine: return True

    try:
        print("[TTS] Initializing engine...")
        tts_engine = pyttsx3.init()
        if not tts_engine: raise Exception("pyttsx3.init() returned None")

        tts_engine.setProperty('rate', tts_rate.get())
        tts_engine.setProperty('volume', 0.9)
        if tts_voice_id.get():
            tts_engine.setProperty('voice', tts_voice_id.get())

        tts_initialized_successfully = True
        print("[TTS] Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"[TTS] Error initializing engine: {e}")
        tts_engine = None
        tts_initialized_successfully = False
        return False

def get_available_voices():
    """Returns a list of available TTS voices (name, id)."""
    temp_engine = None
    try:
        temp_engine = pyttsx3.init()
        if not temp_engine: return []
        voices = temp_engine.getProperty('voices')
        voice_list = [(v.name[:30] + "..." if len(v.name) > 30 else v.name, v.id) for v in voices]
        temp_engine.stop()
        del temp_engine
        return voice_list
    except Exception as e:
        print(f"[TTS] Error getting voices: {e}")
        if temp_engine:
            try: temp_engine.stop()
            except: pass
            del temp_engine
        return []

def set_voice(event=None):
    """Sets the selected voice for the TTS engine."""
    global tts_engine, tts_voice_id, tts_initialized_successfully
    if not tts_initialized_successfully or not tts_engine: return
    try:
        selected_idx = voice_selector.current()
        if selected_idx >= 0:
            voice_id = available_voices[selected_idx][1]
            tts_voice_id.set(voice_id)
            tts_engine.setProperty('voice', voice_id)
            print(f"[TTS] Voice set to: {available_voices[selected_idx][0]}")
    except Exception as e:
        print(f"[TTS] Error setting voice: {e}")

def set_speech_rate(value=None):
    """Sets the speech rate for the TTS engine."""
    global tts_engine, tts_rate, tts_initialized_successfully
    if not tts_initialized_successfully or not tts_engine: return
    try:
        rate = int(float(value))
        tts_rate.set(rate)
        tts_engine.setProperty('rate', rate)
    except Exception as e:
        print(f"[TTS] Error setting speech rate: {e}")

def tts_worker():
    """Worker thread processing the TTS queue."""
    global tts_engine, tts_queue, tts_busy, tts_initialized_successfully
    print("[TTS Worker] Thread started.")
    while True:
        try:
            text_to_speak = tts_queue.get()
            if text_to_speak is None:
                print("[TTS Worker] Received stop signal.")
                break

            if tts_engine and tts_enabled.get() and tts_initialized_successfully:
                with tts_busy_lock:
                    tts_busy = True

                if tts_voice_id.get():
                    tts_engine.setProperty('voice', tts_voice_id.get())
                tts_engine.setProperty('rate', tts_rate.get())

                tts_engine.say(text_to_speak)
                tts_engine.runAndWait()

                with tts_busy_lock:
                    tts_busy = False
            else:
                pass

            tts_queue.task_done()
        except Exception as e:
            print(f"[TTS Worker] Error: {e}")
            with tts_busy_lock:
                tts_busy = False
    print("[TTS Worker] Thread finished.")

def start_tts_thread():
    """Starts the TTS worker thread if needed."""
    global tts_thread, tts_initialized_successfully
    if tts_thread is None or not tts_thread.is_alive():
        if not tts_initialized_successfully:
            tts_initialized_successfully = initialize_tts()

        if tts_initialized_successfully:
            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            tts_thread.start()
            print("[TTS] Worker thread started.")
        else:
            print("[TTS] Engine init failed. Cannot start TTS thread.")

def stop_tts_thread():
    """Signals the TTS worker thread to stop and cleans up."""
    global tts_thread, tts_engine, tts_queue
    print("[TTS] Stopping worker thread...")
    if tts_engine:
        try: tts_engine.stop()
        except Exception as e: print(f"[TTS] Error stopping engine: {e}")

    while not tts_queue.empty():
        try: tts_queue.get_nowait()
        except queue.Empty: break

    if tts_thread and tts_thread.is_alive():
        tts_queue.put(None)
        tts_thread.join(timeout=2)
        if tts_thread.is_alive():
            print("[TTS] Warning: Worker thread did not terminate gracefully.")
    tts_thread = None
    print("[TTS] Worker thread stopped.")

def toggle_tts():
    """Callback for the TTS enable/disable checkbox."""
    global tts_enabled, tts_initialized_successfully
    if tts_enabled.get():
        if not tts_initialized_successfully:
            tts_initialized_successfully = initialize_tts()

        if tts_initialized_successfully:
            print("[TTS] Enabled by user.")
            start_tts_thread()
            if 'voice_selector' in globals(): voice_selector.config(state="readonly")
            if 'rate_scale' in globals(): rate_scale.config(state="normal")
        else:
            print("[TTS] Enable failed - Engine initialization problem.")
            tts_enabled.set(False)
            add_message_to_ui("error", "TTS Engine failed to initialize. Cannot enable TTS.")
            if 'voice_selector' in globals(): voice_selector.config(state="disabled")
            if 'rate_scale' in globals(): rate_scale.config(state="disabled")
    else:
        print("[TTS] Disabled by user.")
        if tts_engine:
            try: tts_engine.stop()
            except Exception as e: print(f"[TTS] Error stopping on toggle off: {e}")
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break


# ========================
# Whisper & VAD Setup
# ========================
def initialize_whisper():
    """Initializes the Whisper model. Returns True on success."""
    global whisper_model, whisper_initialized, whisper_model_size
    if whisper_initialized: return True

    update_vad_status(f"Loading Whisper ({whisper_model_size})...", "blue")
    try:
        print(f"[Whisper] Initializing model ({whisper_model_size})...")

        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

        if whisper_model_size.startswith("turbo"):
            whisper_turbo_model = whisper_model_size.split("-")[1]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            whisper_model = faster_whisper.WhisperModel(whisper_turbo_model, device=device, compute_type=compute_type)
            print(f"[Whisper] Turbo ({whisper_turbo_model}) model loaded on {device} using {compute_type}")
        else:
            whisper_model = whisper.load_model(whisper_model_size)

        whisper_initialized = True
        update_vad_status(f"Whisper ({whisper_model_size}) ready.", "green")
        print("[Whisper] Model initialized successfully.")
        return True
    
    except Exception as e:
        print(f"[Whisper] Error initializing model: {e}")
        whisper_initialized = False
        whisper_model = None
        update_vad_status("Whisper init failed!", "red")
        add_message_to_ui("error", f"Failed to load Whisper {whisper_model_size} model: {e}")
        return False

def initialize_vad():
    """Initializes the Silero VAD model. Returns True on success."""
    global vad_model, vad_utils, vad_get_speech_ts, vad_initialized
    if vad_initialized: return True

    update_vad_status("Loading VAD model...", "blue")
    print("[VAD] Initializing Silero VAD model...")
    
    try:
        torch_hub_dir = os.path.join(tempfile.gettempdir(), "torch_hub_cache")
        os.makedirs(torch_hub_dir, exist_ok=True)
        torch.hub.set_dir(torch_hub_dir)
        print(f"[VAD] Using torch hub cache directory: {torch_hub_dir}")
        vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
    except Exception as e:
        print(f"[VAD] Initial load failed: {e}. Attempting to force reload...")
        try:
            vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, trust_repo=True)
        except Exception as e_reload:
            error_message = (f"Failed to load Silero VAD model even after forcing a reload.\n\nError: {e_reload}\n\n"
                             "This is often caused by an antivirus blocking the download or a network issue.")
            print(f"[VAD] Error initializing model: {error_message}")
            vad_initialized = False
            vad_model = None
            update_vad_status("VAD init failed!", "red")
            add_message_to_ui("error", error_message)
            return False

    (vad_get_speech_ts, _, _, _, _) = vad_utils
    vad_initialized = True
    print("[VAD] Model initialized successfully.")
    return True

def initialize_audio_system():
    """Initializes PyAudio. Returns True on success."""
    global py_audio
    if py_audio: return True
    try:
        print("[Audio] Initializing PyAudio...")
        py_audio = pyaudio.PyAudio()
        print("[Audio] PyAudio initialized.")
        return True
    except Exception as e:
        print(f"[Audio] Error initializing PyAudio: {e}")
        add_message_to_ui("error", f"Failed to initialize audio system: {e}")
        py_audio = None
        return False

def vad_worker():
    """Worker thread for continuous VAD and triggering recording."""
    global py_audio, audio_stream, vad_audio_buffer, audio_frames_buffer, is_recording_for_whisper
    global vad_model, vad_get_speech_ts, vad_stop_event, temp_audio_file_path, whisper_queue
    global tts_busy, tts_busy_lock

    print("[VAD Worker] Thread started.")
    stream = None
    try:
        stream = py_audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=VAD_CHUNK)
        print("[VAD Worker] Audio stream opened.")
        update_vad_status("Listening...", "gray")

        silence_frame_limit = int(SILENCE_THRESHOLD_SECONDS * RATE / VAD_CHUNK)
        pre_speech_buffer_frames = int(PRE_SPEECH_BUFFER_SECONDS * RATE / VAD_CHUNK)
        temp_pre_speech_buffer = deque(maxlen=pre_speech_buffer_frames)
        was_tts_busy = False
        frames_since_last_speech = 0

        while not vad_stop_event.is_set():
            try:
                data = stream.read(VAD_CHUNK, exception_on_overflow=False)
                with tts_busy_lock:
                    current_tts_busy = tts_busy
                
                if current_tts_busy:
                    if is_recording_for_whisper:
                        print("[VAD Worker] Canceling recording due to TTS activity")
                        is_recording_for_whisper = False
                        audio_frames_buffer.clear()
                    if not was_tts_busy:
                        update_vad_status("VAD Paused (TTS Active)", "blue")
                        was_tts_busy = True
                    continue
                
                if was_tts_busy and not current_tts_busy:
                    update_vad_status("Listening...", "gray")
                    was_tts_busy = False
                
                audio_chunk_np = np.frombuffer(data, dtype=np.int16)
                vad_audio_buffer.append(audio_chunk_np)
                temp_pre_speech_buffer.append(data)

                if len(vad_audio_buffer) >= int(RATE / VAD_CHUNK * 0.5):
                    audio_data_np = np.concatenate(list(vad_audio_buffer))
                    audio_float32 = audio_data_np.astype(np.float32) / 32768.0
                    audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0)
                    speech_timestamps = vad_get_speech_ts(audio_tensor, vad_model, sampling_rate=RATE, threshold=0.4)

                    if speech_timestamps:
                        frames_since_last_speech = 0
                        reset_user_activity_timer()
                        if not is_recording_for_whisper:
                            print("[VAD Worker] Speech started, beginning recording.")
                            is_recording_for_whisper = True
                            audio_frames_buffer.clear()
                            for frame_data in temp_pre_speech_buffer:
                                audio_frames_buffer.append(frame_data)
                            update_vad_status("Recording...", "red")
                        audio_frames_buffer.append(data)
                    else:
                        frames_since_last_speech += 1
                        if is_recording_for_whisper and frames_since_last_speech > silence_frame_limit:
                            print(f"[VAD Worker] Silence detected ({SILENCE_THRESHOLD_SECONDS}s), stopping recording.")
                            recording_duration = len(audio_frames_buffer) * VAD_CHUNK / RATE
                            
                            if recording_duration < 0.6 + PRE_SPEECH_BUFFER_SECONDS + SILENCE_THRESHOLD_SECONDS:
                                print(f"[VAD Worker] Recording too short, discarding.")
                                is_recording_for_whisper = False
                                audio_frames_buffer.clear()
                                update_vad_status("Too short, discarded", "orange")
                                window.after(1000, lambda: update_vad_status("Listening...", "gray"))
                            else:
                                temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                                temp_audio_file_path = temp_audio_file.name
                                temp_audio_file.close()

                                with wave.open(temp_audio_file_path, 'wb') as wf:
                                    wf.setnchannels(CHANNELS)
                                    wf.setsampwidth(py_audio.get_sample_size(FORMAT))
                                    wf.setframerate(RATE)
                                    wf.writeframes(b''.join(audio_frames_buffer))
                                print(f"[VAD Worker] Audio saved to {temp_audio_file_path}")
                                whisper_queue.put(temp_audio_file_path)
                                is_recording_for_whisper = False
                                audio_frames_buffer.clear()
                                update_vad_status("Processing...", "blue")
                        
                        if not is_recording_for_whisper:
                             update_vad_status("Listening...", "gray")

            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed: print("[VAD Worker] Warning: Input overflowed.")
                else: print(f"[VAD Worker] Stream read error: {e}"); time.sleep(0.1)
            except Exception as e: print(f"[VAD Worker] Unexpected error: {e}"); time.sleep(0.1)

    except Exception as e:
        print(f"[VAD Worker] Failed to open audio stream: {e}")
        update_vad_status("Audio Error!", "red")
    finally:
        if stream:
            try: stream.stop_stream(); stream.close(); print("[VAD Worker] Audio stream closed.")
            except Exception as e: print(f"[VAD Worker] Error closing stream: {e}")
        is_recording_for_whisper = False
        audio_frames_buffer.clear()
        vad_audio_buffer.clear()
        if not vad_stop_event.is_set(): update_vad_status("VAD Stopped (Error)", "red")
        else: update_vad_status("Voice Disabled", "grey")
    print("[VAD Worker] Thread finished.")

def process_audio_worker():
    """Worker thread to transcribe audio files from the whisper_queue."""
    global whisper_model, whisper_initialized, whisper_queue, whisper_language
    print("[Whisper Worker] Thread started.")
    while True:
        try:
            audio_file_path = whisper_queue.get()
            if audio_file_path is None:
                print("[Whisper Worker] Received stop signal.")
                break

            if not whisper_initialized or not voice_enabled.get():
                print("[Whisper Worker] Skipping transcription (disabled or not initialized).")
                try: os.unlink(audio_file_path)
                except Exception: pass
                whisper_queue.task_done()
                continue

            print(f"[Whisper Worker] Processing audio file: {audio_file_path}")
            update_vad_status("Transcribing...", "orange")
            
            start_time = time.time()
            try:
                lang_to_use = whisper_language if whisper_language else None
                if whisper_model_size.startswith("turbo"):
                    segments, _ = whisper_model.transcribe(audio_file_path, language=lang_to_use)
                    transcribed_text = " ".join(seg.text for seg in segments).strip()
                else:
                    result = whisper_model.transcribe(audio_file_path, language=lang_to_use)
                    transcribed_text = result["text"].strip()

                print(f"[Whisper Worker] Transcription complete in {time.time() - start_time:.2f}s: '{transcribed_text}'")

                if transcribed_text:
                    window.after(0, update_input_with_transcription, transcribed_text)
                    update_vad_status("Transcription Ready", "green")
                else:
                    update_vad_status("No speech detected", "orange")

            except Exception as e:
                print(f"[Whisper Worker] Error during transcription: {e}")
                traceback.print_exc() 
                update_vad_status("Transcription Error", "red")
            finally:
                try: os.unlink(audio_file_path)
                except Exception as e: print(f"[Whisper Worker] Error deleting temp file {audio_file_path}: {e}")
            whisper_queue.task_done()
        except Exception as e:
            print(f"[Whisper Worker] Error: {e}")
    print("[Whisper Worker] Thread finished.")


def update_input_with_transcription(text):
    """Updates the user input text box with the transcribed text."""
    global user_input_widget, auto_clear_on_new_content
    if not user_input_widget: return

    if auto_clear_on_new_content.get():
        user_input_widget.delete("1.0", tk.END)
        user_input_widget.insert(tk.END, text)
    else:
        current_text = user_input_widget.get("1.0", tk.END).strip()
        user_input_widget.insert(tk.END, (" " if current_text else "") + text)
    
    if auto_send_after_transcription.get():
        window.after(100, send_message)


def toggle_voice_recognition():
    """Enables/disables VAD and Whisper."""
    global voice_enabled, whisper_initialized, vad_initialized, vad_thread, vad_stop_event
    global py_audio, whisper_processing_thread

    set_whisper_language()

    if voice_enabled.get():
        print("[Voice] Enabling voice recognition...")
        all_initialized = True
        if not py_audio and not initialize_audio_system(): all_initialized = False
        if not whisper_initialized and not initialize_whisper(): all_initialized = False
        if not vad_initialized and not initialize_vad(): all_initialized = False

        if all_initialized:
            if whisper_processing_thread is None or not whisper_processing_thread.is_alive():
                whisper_processing_thread = threading.Thread(target=process_audio_worker, daemon=True)
                whisper_processing_thread.start()
            if vad_thread is None or not vad_thread.is_alive():
                vad_stop_event.clear()
                vad_thread = threading.Thread(target=vad_worker, daemon=True)
                vad_thread.start()
            update_vad_status("Voice Enabled", "green")
            print("[Voice] Voice recognition enabled.")
        else:
            print("[Voice] Enabling failed due to initialization errors.")
            voice_enabled.set(False)
            update_vad_status("Init Failed", "red")

    else:
        print("[Voice] Disabling voice recognition...")
        update_vad_status("Disabling...", "grey")
        if vad_thread and vad_thread.is_alive():
            vad_stop_event.set()
            vad_thread.join(timeout=2)
            if vad_thread.is_alive(): print("[Voice] Warning: VAD thread did not stop gracefully.")
            vad_thread = None
        print("[Voice] Voice recognition disabled.")

# File Handling (largely unchanged)
def extract_pdf_content(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text_content = ""
            metadata = doc.metadata
            if metadata:
                text_content += f"PDF Title: {metadata.get('title', 'N/A')}\nAuthor: {metadata.get('author', 'N/A')}\n\n"
            for page_num, page in enumerate(doc):
                text_content += f"--- Page {page_num+1} ---\n{page.get_text()}\n\n"
        return text_content
    except Exception as e: return f"Error extracting PDF content: {str(e)}"

def select_file():
    file_path = filedialog.askopenfilename(
        title="Select File",
        filetypes=[("All Supported Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.pdf;*.txt;*.md;*.py"),
                   ("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp"), ("PDF files", "*.pdf"),
                   ("Text files", "*.txt;*.md;*.py;*.js;*.html;*.css;*.json")])
    if file_path:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']: handle_image_file(file_path)
        elif file_ext == '.pdf': handle_pdf_file(file_path)
        elif file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']: handle_text_file(file_path)

def handle_image_file(file_path):
    global selected_image_path, image_sent_in_history
    selected_image_path = file_path
    image_sent_in_history = False
    update_image_preview(file_path)
    image_indicator.config(text=f"✓ {os.path.basename(file_path)}")
    if 'clear_attach_button' in globals():
        clear_attach_button.pack(side=tk.LEFT, after=attach_button, padx=5)

def handle_pdf_file(file_path):
    add_message_to_ui("status", f"Loading PDF: {os.path.basename(file_path)}...")
    threading.Thread(target=lambda: window.after(0, update_input_with_pdf_content, extract_pdf_content(file_path), file_path), daemon=True).start()

def handle_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(10000)
            if len(content) == 10000: content += "\n\n[Content truncated]"
        user_input_widget.delete("1.0", tk.END); user_input_widget.insert("1.0", content)
        add_message_to_ui("status", f"Text loaded from: {os.path.basename(file_path)}")
    except Exception as e: add_message_to_ui("error", f"Error loading text file: {e}")

def update_input_with_pdf_content(content, file_path):
    if len(content) > 10000: content = content[:10000] + "\n\n[Content truncated]"
    user_input_widget.delete("1.0", tk.END); user_input_widget.insert("1.0", content)
    add_message_to_ui("status", f"PDF loaded: {os.path.basename(file_path)}")

def paste_image_from_clipboard(event=None):
    global selected_image_path, image_sent_in_history
    try:
        from PIL import ImageGrab
        clipboard_image = ImageGrab.grabclipboard()
        if not isinstance(clipboard_image, Image.Image):
            add_message_to_ui("status", "No image in clipboard."); return
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name
        clipboard_image.save(temp_path, "PNG")
        selected_image_path = temp_path
        image_sent_in_history = False
        update_image_preview(temp_path)
        image_indicator.config(text="✓ Pasted image")
        if 'clear_attach_button' in globals():
            clear_attach_button.pack(side=tk.LEFT, after=attach_button, padx=5)
    except Exception as e: add_message_to_ui("error", f"Failed to paste image: {e}")

def setup_paste_binding():
    window.bind("<Control-v>", paste_image_from_clipboard)
    user_input_widget.bind("<Control-v>", lambda e: window.after(10, paste_image_from_clipboard))
    window.bind("<Command-v>", paste_image_from_clipboard)
    user_input_widget.bind("<Command-v>", lambda e: window.after(10, paste_image_from_clipboard))

def handle_drop(event, target_widget):
    file_path = event.data.strip('{}')
    if file_path and os.path.isfile(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']: handle_image_file(file_path)
        elif ext == '.pdf': handle_pdf_file(file_path)
        else: handle_text_file(file_path)

def set_whisper_language(event=None):
    global whisper_language
    lang_code = WHISPER_LANGUAGES[whisper_language_selector.current()][1]
    whisper_language = lang_code
    print(f"[Whisper] Language set to: {whisper_language}")

def set_whisper_model_size(event=None):
    global whisper_model_size, whisper_initialized, whisper_model
    new_size = whisper_model_size_selector.get()
    if new_size != whisper_model_size:
        whisper_model_size = new_size
        whisper_initialized = False
        whisper_model = None
        if voice_enabled.get(): initialize_whisper()

def update_vad_status(text, color):
    if recording_indicator_widget and window:
        try: window.after(0, lambda: recording_indicator_widget.config(text=text, fg=color))
        except tk.TclError: pass


# ===================
# Ollama / Chat Logic
# ===================
def fetch_available_models():
    """Fetches available Ollama models, compatible with new and old library versions."""
    try:
        models_data = ollama.list()
        valid_models = []
        if hasattr(models_data, 'models') and isinstance(models_data.models, list): # New format
            for model_obj in models_data.models:
                if hasattr(model_obj, 'model'): valid_models.append(model_obj.model)
        elif isinstance(models_data, dict) and 'models' in models_data: # Old format
            for model in models_data.get('models', []):
                if isinstance(model, dict) and 'name' in model: valid_models.append(model['name'])
        return valid_models if valid_models else [DEFAULT_OLLAMA_MODEL, "None"]
    except Exception as e:
        print(f"[Ollama] Error fetching models: {e}")
        return [DEFAULT_OLLAMA_MODEL, "None"]

def chat_worker(user_message_content, system_prompt_content, image_path=None):
    """Background worker for Ollama streaming chat."""
    global messages, current_model, stream_queue, stream_done_event, stream_in_progress

    current_message = {"role": "user", "content": user_message_content}
    if image_path:
        try:
            with open(image_path, 'rb') as f:
                current_message["images"] = [f.read()]
        except Exception as e:
            stream_queue.put(("ERROR", f"Error reading image file: {e}"))
            stream_in_progress = False
            stream_done_event.set()
            return

    messages.append(current_message)
    history_for_ollama = []
    if system_prompt_content:
        history_for_ollama.append({"role": "system", "content": system_prompt_content})
    history_for_ollama.extend(messages)

    assistant_response = ""
    try:
        print(f"[Ollama] Sending request to model {current_model}...")
        stream = ollama.chat(model=current_model, messages=history_for_ollama, stream=True)
        
        stream_queue.put(("START", None))
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content_piece = chunk['message']['content']
                stream_queue.put(("CHUNK", content_piece))
                assistant_response += content_piece
            if 'error' in chunk:
                stream_queue.put(("ERROR", f"Ollama error: {chunk['error']}"))
                break
        
        if assistant_response:
             messages.append({"role": "assistant", "content": assistant_response})
        
        stream_queue.put(("END", None))

    except Exception as e:
        stream_queue.put(("ERROR", f"Ollama communication error: {e}"))
    finally:
        stream_in_progress = False
        stream_done_event.set()

def process_line(line):
    """Processes a single line from the LLM response buffer."""
    global is_in_think_block, tts_queue, chat_history_widget

    line = line.strip()
    if not line: return

    # Simple state machine for <think> blocks
    if "<think>" in line: is_in_think_block = True
    if is_in_think_block:
        if "</think>" in line:
            is_in_think_block = False
        return # Ignore all content related to thinking

    # Display the line and speak if it's Spanish
    chat_history_widget.config(state=tk.NORMAL)
    if line.startswith("[EN]"):
        translation_text = line[4:].strip()
        chat_history_widget.insert(tk.END, translation_text + "\n", "english_translation")
    else:
        chat_history_widget.insert(tk.END, line + "\n", "spanish_sentence")
        if tts_enabled.get() and tts_initialized_successfully:
            tts_queue.put(line)
    
    chat_history_widget.config(state=tk.DISABLED)
    chat_history_widget.see(tk.END)


def process_stream_queue():
    """Processes items from Ollama stream queue for UI and TTS."""
    global stream_queue, chat_history_widget, stream_buffer, is_in_think_block

    try:
        while True:
            item_type, item_data = stream_queue.get_nowait()

            if item_type == "START":
                remove_thinking_message()
                chat_history_widget.config(state=tk.NORMAL)
                chat_history_widget.insert(tk.END, "______________________________Maestro______________________________\n\n", "bot_tag")
                chat_history_widget.config(state=tk.DISABLED)

            elif item_type == "CHUNK":
                stream_buffer += item_data
                while '\n' in stream_buffer:
                    line, stream_buffer = stream_buffer.split('\n', 1)
                    process_line(line)

            elif item_type == "END":
                if stream_buffer:
                    process_line(stream_buffer)
                    stream_buffer = ""
                chat_history_widget.config(state=tk.NORMAL)
                chat_history_widget.insert(tk.END, "\n")
                chat_history_widget.config(state=tk.DISABLED)
                chat_history_widget.see(tk.END)
                is_in_think_block = False

                def wait_for_tts_and_reset():
                    """Polls to see if TTS is done, then resets the inactivity timer."""
                    with tts_busy_lock:
                        is_busy = tts_busy
                    
                    if not tts_queue.empty() or is_busy:
                        window.after(200, wait_for_tts_and_reset)
                    else:
                        print("[Inactivity] TTS complete. Resetting user inactivity timer.")
                        reset_user_activity_timer()
                
                if tts_enabled.get() and tts_initialized_successfully:
                    wait_for_tts_and_reset()
                else:
                    reset_user_activity_timer()
                
                return

            elif item_type == "ERROR":
                remove_thinking_message()
                add_message_to_ui("error", item_data)
                global stream_in_progress
                stream_in_progress = False
                reset_user_activity_timer()
                return

    except queue.Empty:
        pass

    if stream_in_progress:
        window.after(50, process_stream_queue)


# ===================
# UI Helpers
# ===================

def set_initial_window_geometry(root):
    """Calculates and sets the window size and position to fill the work area vertically."""
    window_width = 1000  # Keep the original width

    # Use a temporary Toplevel window to find the work area
    # This is a common method to get the screen size without the taskbar
    temp = tk.Toplevel(root)
    temp.attributes("-alpha", 0.0)  # Make it invisible
    
    # Try to maximize the window to get work area
    try:
        temp.state('zoomed')
        temp.update_idletasks()
        work_area_height = temp.winfo_height() - 10 # Subtract a small margin for easier sizing
        screen_width = temp.winfo_screenwidth()
    except tk.TclError:
        # Fallback if 'zoomed' is not supported (e.g., some Linux WMs)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        # Assume a standard taskbar height as a fallback
        work_area_height = screen_height - 80 
        print("Warning: Could not determine taskbar height automatically. Using a fallback.")
    finally:
        temp.destroy()
    
    # Calculate coordinates to center horizontally and align to the top
    x_coordinate = (screen_width - window_width) // 2
    y_coordinate = 0  # Align to the top of the screen

    # Set the geometry
    root.geometry(f'{window_width}x{work_area_height}+{x_coordinate}+{y_coordinate}')

def clear_image():
    global selected_image_path, image_sent_in_history
    selected_image_path = ""
    image_sent_in_history = False
    image_preview.configure(image="", text="", width=0, height=0, bg="#F8F8F8")
    image_preview.image = None
    image_indicator.config(text="")
    attachment_strip.config(height=0)
    if 'clear_attach_button' in globals():
        clear_attach_button.pack_forget()

def update_image_preview(file_path):
    try:
        img = Image.open(file_path)
        img.thumbnail((80, 80), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        image_preview.configure(image=photo, width=img.width, height=img.height, text="")
        image_preview.image = photo
        image_preview.pack(side=tk.LEFT, padx=5)
        image_indicator.config(text=f"📎 {os.path.basename(file_path)}")
        image_indicator.pack(side=tk.LEFT, padx=5)
        attachment_strip.config(height=30)
        attachment_strip.pack(fill=tk.X, pady=(5,0))
        if 'clear_attach_button' in globals():
            clear_attach_button.pack(side=tk.LEFT, after=attach_button, padx=5)
    except Exception as e:
        clear_image()
        image_indicator.config(text="Preview Error", fg="red")

def add_message_to_ui(role, content, tag_suffix=""):
    if not chat_history_widget or not content: return
    chat_history_widget.config(state=tk.NORMAL)
    if role == "user":
        chat_history_widget.insert(tk.END, "________________________________You________________________________ \n\n", "user_tag")
        chat_history_widget.insert(tk.END, content + "\n\n", "user_message"+tag_suffix)
    elif role == "assistant": # This role is now handled by streaming
        pass
    elif role == "error":
        chat_history_widget.insert(tk.END, f"Error: {content}\n\n", "error"+tag_suffix)
    elif role == "status":
         chat_history_widget.insert(tk.END, f"{content}\n\n", "status"+tag_suffix)
    chat_history_widget.see(tk.END)
    chat_history_widget.config(state=tk.DISABLED)
    if 'window' in globals() and window:
         try: window.update_idletasks()
         except tk.TclError: pass

def remove_thinking_message():
    """Removes the 'Thinking...' indicator from the chat."""
    global thinking_start_index
    if thinking_start_index:
        try:
            chat_history_widget.config(state=tk.NORMAL)
            thinking_text_content = "Thinking...\n"
            end_thinking = f"{thinking_start_index}+{len(thinking_text_content)}c"
            
            if chat_history_widget.get(thinking_start_index, end_thinking) == thinking_text_content:
                chat_history_widget.delete(thinking_start_index, end_thinking)
            
            chat_history_widget.config(state=tk.DISABLED)
        except tk.TclError as e:
            print(f"UI Error removing 'Thinking...' message (might be benign): {e}")
        finally:
            thinking_start_index = None

def select_model(event=None):
    global current_model
    selected = model_selector.get()
    if selected and selected != "No models found":
        current_model = selected
        model_status.config(text=f"Using: {current_model.split(':')[0]}")
        print(f"[Ollama] Model selected: {current_model}")

def send_message(event=None):
    global messages, selected_image_path, image_sent_in_history, stream_in_progress, stream_done_event, thinking_start_index


    if stream_in_progress:
        add_message_to_ui("error", "Please wait for the current response to complete.")
        return

    user_text = user_input_widget.get("1.0", tk.END).strip()
    image_to_send = selected_image_path if not image_sent_in_history else None
    system_prompt_text = system_prompt_input_widget.get("1.0", tk.END).strip()

    if not user_text and not image_to_send:
        add_message_to_ui("error", "Please enter a message or attach a new image.")
        return

    send_button.config(state=tk.DISABLED)
    stream_in_progress = True
    stream_done_event.clear()

    display_text = user_text + (f" [Image: {os.path.basename(image_to_send)}]" if image_to_send else "")
    add_message_to_ui("user", display_text if display_text else "[Image Attached]")
    user_input_widget.delete("1.0", tk.END)

    # Add "Thinking" indicator
    chat_history_widget.config(state=tk.NORMAL)
    thinking_start_index = chat_history_widget.index(tk.INSERT)
    chat_history_widget.insert(tk.INSERT, "Thinking...\n", "thinking")
    chat_history_widget.see(tk.END)
    chat_history_widget.config(state=tk.DISABLED)
    window.update_idletasks()

    if tts_engine and tts_enabled.get():
        try: tts_engine.stop()
        except: pass
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break
        with tts_busy_lock: tts_busy = False

    thread = threading.Thread(target=chat_worker, args=(user_text, system_prompt_text, image_to_send), daemon=True)
    thread.start()

    if image_to_send:
        image_sent_in_history = True

    def check_done():
        if stream_done_event.is_set():
            send_button.config(state=tk.NORMAL)
        else:
            window.after(100, check_done)

    window.after(50, process_stream_queue)
    window.after(100, check_done)


# ===================
# UI Helper Functions
# ===================

def toggle_frame(frame, button=None):
    """Toggle the visibility of a frame and update button text if provided."""
    if frame.winfo_viewable():
        frame.pack_forget()
        if button:
            current_text = button.cget("text")
            button.config(text=current_text.replace("▾", "▸"))
    else:
        frame.pack(fill=tk.X, pady=(5, 15))
        if button:
            current_text = button.cget("text")
            button.config(text=current_text.replace("▸", "▾"))

# ===================
# Main Application Setup & Loop
# ===================
def on_closing():
    """Handles application shutdown gracefully."""
    print("[Main] Closing application...")
    if vad_thread and vad_thread.is_alive():
        vad_stop_event.set(); vad_thread.join(timeout=2)
    if whisper_processing_thread and whisper_processing_thread.is_alive():
        whisper_queue.put(None); whisper_processing_thread.join(timeout=2)
    stop_tts_thread()
    if py_audio: py_audio.terminate()
    if temp_audio_file_path and os.path.exists(temp_audio_file_path):
        try: os.unlink(temp_audio_file_path)
        except Exception as e: print(f"Error deleting temp file: {e}")
    window.destroy()
    print("[Main] Application closed.")


# --- Build GUI ---
window = TkinterDnD.Tk()
window.title(APP_TITLE)
set_initial_window_geometry(window)


# --- Tkinter Variables ---
tts_enabled = tk.BooleanVar(value=True)
tts_rate = tk.IntVar(value=175) # Slightly slower default for language learning
tts_voice_id = tk.StringVar(value="")
voice_enabled = tk.BooleanVar(value=True)
auto_send_after_transcription = tk.BooleanVar(value=True)
auto_clear_on_new_content = tk.BooleanVar(value=False)
selected_whisper_language = tk.StringVar()
selected_whisper_model = tk.StringVar(value=whisper_model_size)
inactivity_enabled = tk.BooleanVar(value=True)
inactivity_threshold_seconds = tk.IntVar(value=60)  # Default 60 seconds (matches USER_INACTIVITY_THRESHOLD/1000)
continuation_prompt = tk.StringVar(value=DEFAULT_CONTINUATION_PROMPT)

# --- Main Frame ---
main_frame = tk.Frame(window, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# --- Create Main Paned Layout ---
paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
paned_window.pack(fill=tk.BOTH, expand=True)

# Create the two main frames
controls_sidebar_frame = tk.Frame(paned_window, padx=10, pady=10, bg="#F5F5F7")
conversation_frame = tk.Frame(paned_window, padx=10, pady=10)

# Add them to the paned window
paned_window.add(controls_sidebar_frame, weight=1)
paned_window.add(conversation_frame, weight=3)

# Set the initial sash position (sidebar width)
window.after(100, lambda: paned_window.sashpos(0, 350))

# --- Model Selection ---
model_heading = tk.Label(controls_sidebar_frame, text="Model", font=("Arial", 12, "bold"), anchor="w", bg="#F5F5F7")
model_heading.pack(fill=tk.X, pady=(0, 5))

model_selector_frame = tk.Frame(controls_sidebar_frame, bg="#F5F5F7")
model_selector_frame.pack(fill=tk.X, expand=False, pady=(0, 15))

available_models = fetch_available_models()
model_selector = ttk.Combobox(model_selector_frame, values=available_models, state="readonly")
if available_models:
    model_to_set = next((m for m in [current_model, DEFAULT_OLLAMA_MODEL] if m in available_models), available_models[0])
    model_selector.set(model_to_set)
    current_model = model_to_set
else:
    model_selector.set("No models found"); model_selector.config(state=tk.DISABLED); current_model = None
model_selector.pack(fill=tk.X, expand=True, pady=(0,2))
model_selector.bind("<<ComboboxSelected>>", select_model)

model_status = tk.Label(model_selector_frame, text=f"Using: {current_model.split(':')[0] if current_model else 'N/A'}", font=("Arial", 8), anchor="w", bg="#F5F5F7")
model_status.pack(fill=tk.X)

# --- TTS Controls (Collapsible) ---
tts_header_button = ttk.Button(
    controls_sidebar_frame,
    text="Text-to-Speech ▸",
    command=lambda: toggle_frame(tts_details_frame, tts_header_button)
)
tts_header_button.pack(fill=tk.X, pady=(0, 5))

tts_details_frame = tk.Frame(controls_sidebar_frame, bg="#F5F5F7")
# Don't pack by default - collapsed

tts_toggle_button = ttk.Checkbutton(tts_details_frame, text="Enable TTS", variable=tts_enabled, command=toggle_tts)
tts_toggle_button.pack(anchor="w", pady=(0, 10))

# --- Voice Selector ---
tk.Label(tts_details_frame, text="Voice:", font=("Arial", 8), bg="#F5F5F7", anchor="w").pack(fill=tk.X)
available_voices = get_available_voices()
voice_names = [v[0] for v in available_voices]
voice_selector = ttk.Combobox(tts_details_frame, values=voice_names, state="disabled", font=("Arial", 8))
if available_voices:
    default_voice_index = 0
    spanish_keywords = ["spanish", "español", "sabina", "helena", "jorge", "elena"]
    for i, (name, v_id) in enumerate(available_voices):
        if any(keyword in name.lower() for keyword in spanish_keywords):
            default_voice_index = i; break
    else:
        for i, (name, v_id) in enumerate(available_voices):
            if any(common in name.lower() for common in ["david", "zira", "hazel", "susan"]):
                default_voice_index = i; break
    voice_selector.current(default_voice_index)
    tts_voice_id.set(available_voices[default_voice_index][1])
    voice_selector.bind("<<ComboboxSelected>>", set_voice)
voice_selector.pack(fill=tk.X, pady=(2, 10))

# --- Rate Slider ---
rate_title_frame = tk.Frame(tts_details_frame, bg="#F5F5F7")
rate_title_frame.pack(fill=tk.X)
tk.Label(rate_title_frame, text="Talking speed:", font=("Arial", 8), bg="#F5F5F7").pack(side=tk.LEFT)
tk.Label(rate_title_frame, textvariable=tts_rate, font=("Arial", 8), bg="#F5F5F7").pack(side=tk.RIGHT)

rate_scale = ttk.Scale(tts_details_frame, from_=80, to=300, orient=tk.HORIZONTAL, variable=tts_rate, command=set_speech_rate, state="disabled")
rate_scale.pack(fill=tk.X, pady=(2, 5))


# --- Voice Recognition Controls (Collapsible) ---
voice_header_button = ttk.Button(
    controls_sidebar_frame,
    text="Voice Input ▸",
    command=lambda: toggle_frame(voice_details_frame, voice_header_button)
)
voice_header_button.pack(fill=tk.X, pady=(0, 5))

voice_details_frame = tk.Frame(controls_sidebar_frame, bg="#F5F5F7")
# Don't pack by default - collapsed

voice_toggle_button = ttk.Checkbutton(voice_details_frame, text="Enable Voice", variable=voice_enabled, command=toggle_voice_recognition)
voice_toggle_button.pack(anchor="w", pady=(0, 10))

# --- Language Selector ---
tk.Label(voice_details_frame, text="Spoken Language:", font=("Arial", 8), bg="#F5F5F7", anchor="w").pack(fill=tk.X)
whisper_language_selector = ttk.Combobox(voice_details_frame, values=[lang[0] for lang in WHISPER_LANGUAGES], state="readonly", font=("Arial", 8), textvariable=selected_whisper_language)
whisper_language_selector.current(0) # Default to Auto Detect
whisper_language_selector.pack(fill=tk.X, pady=(2, 10))
whisper_language_selector.bind("<<ComboboxSelected>>", set_whisper_language)

# --- Model Size Selector ---
tk.Label(voice_details_frame, text="Whisper Model Size:", font=("Arial", 8), bg="#F5F5F7", anchor="w").pack(fill=tk.X)
whisper_model_size_selector = ttk.Combobox(voice_details_frame, values=WHISPER_MODEL_SIZES, state="readonly", font=("Arial", 8), textvariable=selected_whisper_model)
whisper_model_size_selector.pack(fill=tk.X, pady=(2, 10))
whisper_model_size_selector.bind("<<ComboboxSelected>>", set_whisper_model_size)

# --- Options ---
auto_send_checkbox = ttk.Checkbutton(voice_details_frame, text="Auto-send after transcription", variable=auto_send_after_transcription)
auto_send_checkbox.pack(anchor="w")
auto_clear_checkbox = ttk.Checkbutton(voice_details_frame, text="Replace text with new transcription", variable=auto_clear_on_new_content)
auto_clear_checkbox.pack(anchor="w", pady=(0, 5))

# --- Status Indicator ---
recording_indicator_widget = tk.Label(voice_details_frame, text="Voice Disabled", font=("Arial", 9, "italic"), fg="grey", anchor="w", bg="#F5F5F7")
recording_indicator_widget.pack(fill=tk.X, pady=(5,0))


# --- System Prompt Frame (Collapsible) ---
system_prompt_header_button = ttk.Button(
    controls_sidebar_frame,
    text="System Prompt ▸",
    command=lambda: toggle_frame(system_prompt_details_frame, system_prompt_header_button)
)
system_prompt_header_button.pack(fill=tk.X, pady=(0, 5))

system_prompt_details_frame = tk.Frame(controls_sidebar_frame, bg="#F5F5F7")
# Don't pack by default - collapsed

# --- Saved Prompts Dropdown ---
tk.Label(system_prompt_details_frame, text="Saved Prompts:", anchor="w", font=("Arial", 8), bg="#F5F5F7").pack(fill=tk.X)
system_prompt_selector = ttk.Combobox(system_prompt_details_frame, state="readonly", font=("Arial", 8))
system_prompt_selector.pack(fill=tk.X, pady=2)
system_prompt_selector.bind("<<ComboboxSelected>>", on_system_prompt_selected)

# --- Dropdown Action Buttons ---
dropdown_actions_frame = tk.Frame(system_prompt_details_frame, bg="#F5F5F7")
dropdown_actions_frame.pack(fill=tk.X, pady=(2, 10))
apply_button = tk.Button(dropdown_actions_frame, text="Apply Selected", command=apply_system_prompt, font=("Arial", 8), relief=tk.FLAT)
apply_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
delete_button = tk.Button(dropdown_actions_frame, text="Delete Selected", command=delete_selected_system_prompt, font=("Arial", 8), relief=tk.FLAT)
delete_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

# --- Prompt Editor ---
prompt_status = tk.Label(system_prompt_details_frame, text="Active: Default system prompt", 
                         font=("Arial", 8, "italic"), anchor="w", fg="green", bg="#F5F5F7")
prompt_status.pack(fill=tk.X)

system_prompt_input_widget = scrolledtext.ScrolledText(system_prompt_details_frame, wrap=tk.WORD, height=6, font=("Arial", 9))
system_prompt_input_widget.pack(fill=tk.BOTH, expand=True, pady=2)
system_prompt_input_widget.insert("1.0", DEFAULT_SYSTEM_PROMPT)
system_prompt_input_widget.bind("<<Modified>>", lambda e: (on_text_change(), system_prompt_input_widget.edit_modified(False)))

# --- Editor Action Buttons ---
editor_actions_frame = tk.Frame(system_prompt_details_frame, bg="#F5F5F7")
editor_actions_frame.pack(fill=tk.X, pady=(5,0))
save_button = tk.Button(editor_actions_frame, text="Save Current Text as New...", command=save_current_system_prompt, font=("Arial", 8, "bold"), relief=tk.FLAT, bg="#d0e0ff")
save_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
clear_button = tk.Button(editor_actions_frame, text="Reset to Default", command=clear_system_prompt_fields, font=("Arial", 8), relief=tk.FLAT)
clear_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

# --- Inactivity Settings Frame ---
inactivity_settings_frame = create_inactivity_settings_ui()

# --- Chat History ---
chat_history_widget = scrolledtext.ScrolledText(conversation_frame, wrap=tk.WORD, height=15, state=tk.DISABLED, font=("Arial", 10), relief=tk.FLAT, bd=0)
chat_history_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

# Define text tags with improved typography
chat_history_widget.tag_config("user_tag", foreground="#8A8A8E", font=("Arial", 9, "bold"), spacing1=10)
chat_history_widget.tag_config("user_message", foreground="black", spacing3=15)
chat_history_widget.tag_config("bot_tag", foreground="#8A8A8E", font=("Arial", 9, "bold"), spacing1=10)
chat_history_widget.tag_config("thinking", foreground="gray", font=("Arial", 10, "italic"))
chat_history_widget.tag_config("error", foreground="red", font=("Arial", 10, "bold"))
chat_history_widget.tag_config("status", foreground="purple", font=("Arial", 9, "italic"))
chat_history_widget.tag_config("spanish_sentence", font=("Arial", 14), foreground="#000000", spacing3=2)
chat_history_widget.tag_config("english_translation", font=("Arial", 11, "italic"), foreground="#6C6C70", spacing1=0, spacing3=20, lmargin1=15, lmargin2=15)

# --- Input Area (Redesigned for conversation frame) ---
# Place this inside the conversation_frame for a more integrated design

# Main input container with subtle styling
input_container = tk.Frame(conversation_frame, bg="white", relief=tk.FLAT, bd=1, highlightbackground="#E5E5E5", highlightthickness=1)
input_container.pack(fill=tk.X, side=tk.BOTTOM, pady=(10,0))

# Attachment indicator strip (appears above text input when file is attached)
attachment_strip = tk.Frame(input_container, bg="#F8F8F8", height=0)
attachment_strip.pack(fill=tk.X, pady=(0,0))

# Initially hidden attachment preview
image_preview = tk.Label(attachment_strip, text="", bg="#F8F8F8", font=("Arial", 8), fg="grey")
image_indicator = tk.Label(attachment_strip, text="", font=("Arial", 8, "italic"), fg="grey", bg="#F8F8F8")

# Text input area with no border of its own
user_input_widget = scrolledtext.ScrolledText(input_container, wrap=tk.WORD, height=4, font=("Arial", 12), relief=tk.FLAT, bd=0, bg="white")
user_input_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
user_input_widget.focus_set()
user_input_widget.drop_target_register(DND_FILES)
user_input_widget.dnd_bind('<<Drop>>', lambda e: handle_drop(e, user_input_widget))

# Action buttons row
action_button_frame = tk.Frame(input_container, bg="white")
action_button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

# Attach button on the left
attach_button = tk.Button(action_button_frame, text="📎 Attach", command=select_file, relief=tk.FLAT, 
                         font=("Arial", 9), bg="#F0F0F0", fg="black", padx=10)
attach_button.pack(side=tk.LEFT)

# Clear attachment button (initially hidden)
clear_attach_button = tk.Button(action_button_frame, text="✕", command=clear_image, relief=tk.FLAT,
                               font=("Arial", 8), bg="#F0F0F0", fg="red", width=3)

# Send button on the right with Apple blue styling
send_button = tk.Button(action_button_frame, text="Send", command=send_message, relief=tk.FLAT,
                       font=("Arial", 10, "bold"), bg="#007AFF", fg="white", padx=20, pady=5)
send_button.pack(side=tk.RIGHT)
send_button.pack(pady=5, anchor='e')

user_input_widget.bind("<KeyPress-Return>", lambda e: "break" if not (e.state & 0x0001) and send_message() else None)

# --- Final Setup and Initialization ---
window.protocol("WM_DELETE_WINDOW", on_closing)
load_system_prompts()
populate_system_prompt_selector()
load_continuation_prompts()
populate_continuation_prompt_selector()
print("[Main] Pre-initializing TTS...")
if initialize_tts():
    voice_selector.config(state="readonly")
    rate_scale.config(state="normal")
else:
    window.after(500, lambda: add_message_to_ui("status", "Note: TTS engine failed to initialize."))

window.after(1000, toggle_tts)
window.after(1500, toggle_voice_recognition)
setup_paste_binding()

user_input_widget.bind("<KeyRelease>", lambda e: reset_user_activity_timer())
# Start the inactivity checker
reset_user_activity_timer() # Initialize the timer
window.after(USER_INACTIVITY_CHECK_INTERVAL, check_user_inactivity)

# --- Start Main Loop ---
window.mainloop()