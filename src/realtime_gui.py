"""
Real-time Language Recognition GUI

This module provides a graphical user interface for real-time language recognition
using Whisper model and microphone input or audio file selection.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from datetime import datetime
from typing import Optional
import yaml

from src.whisper_recognizer import WhisperLanguageRecognizer, ModelLoadError, WhisperRecognizerError
from src.groq_recognizer import GroqLanguageRecognizer, GroqRecognizerError
from src.audio_stream_handler import AudioStreamHandler, MicrophoneError, AudioStreamError
from src.session_history import SessionHistory, DetectionResult


class RealtimeLanguageGUI:
    """
    GUI application for real-time language recognition.
    
    Provides interface for:
    - Real-time language detection from microphone
    - Language detection from audio files
    - Detection history tracking and export
    """
    
    def __init__(self):
        """Initialize the GUI application and all components."""
        # Load configuration
        self.config = self._load_config()
        
        # Initialize main window with modern styling
        self.root = tk.Tk()
        self.root.title("Language Recognition")
        self.root.geometry("700x800")
        self.root.resizable(False, False)
        
        # Modern color scheme
        self.bg_color = "#ecf0f1"
        self.primary_color = "#3498db"
        self.success_color = "#27ae60"
        self.danger_color = "#e74c3c"
        self.warning_color = "#f39c12"
        self.dark_color = "#2c3e50"
        
        self.root.configure(bg=self.bg_color)
        
        # Initialize components
        self.whisper_recognizer: Optional[WhisperLanguageRecognizer] = None
        self.groq_recognizer: Optional[GroqLanguageRecognizer] = None
        self.audio_handler: Optional[AudioStreamHandler] = None
        self.session_history = SessionHistory()
        
        # State variables
        self.is_processing = False
        self.use_groq = False  # Flag to switch between Whisper and Groq
        
        # Initialize recognizers
        self._initialize_recognizers()
        
        # Setup GUI layout
        self._setup_gui()
        
    def _load_config(self) -> dict:
        """Load configuration from config.yaml."""
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}")
            return {
                'whisper': {
                    'model_size': 'small',
                    'chunk_duration': 2.0,
                    'sample_rate': 16000,
                    'silence_threshold': 0.005,
                    'min_confidence': 0.25
                }
            }
    
    def _initialize_recognizers(self):
        """Initialize both Whisper and Groq recognizers with error handling."""
        # Try to initialize Groq first (faster, cloud-based)
        groq_api_key = self.config.get('groq', {}).get('api_key', '')
        if groq_api_key:
            try:
                groq_model = self.config.get('groq', {}).get('model', 'whisper-large-v3')
                self.groq_recognizer = GroqLanguageRecognizer(api_key=groq_api_key, model=groq_model)
                self.use_groq = True
                print("Groq API initialized successfully")
            except GroqRecognizerError as e:
                print(f"Failed to initialize Groq recognizer: {e}")
                self.groq_recognizer = None
        
        # Try to initialize Whisper as fallback
        if not self.use_groq:
            try:
                model_size = self.config.get('whisper', {}).get('model_size', 'base')
                self.whisper_recognizer = WhisperLanguageRecognizer(model_size=model_size)
                print("Whisper model initialized successfully")
            except ModelLoadError as e:
                print(f"Failed to load Whisper model: {e}")
                self.whisper_recognizer = None
            except Exception as e:
                print(f"Failed to initialize Whisper recognizer: {e}")
                self.whisper_recognizer = None
        
        # Show error if neither is available
        if not self.groq_recognizer and not self.whisper_recognizer:
            messagebox.showerror(
                "Initialization Error",
                "Failed to initialize any language recognizer.\n\n"
                "Please either:\n"
                "1. Add Groq API key to config.yaml, or\n"
                "2. Install openai-whisper: pip install openai-whisper"
            )
    
    def _setup_gui(self):
        """Setup the GUI layout and all widgets."""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'))
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=self.bg_color, padx=25, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Modern header with gradient-like effect
        header_frame = tk.Frame(main_frame, bg="#2c3e50", height=100)
        header_frame.pack(fill=tk.X, pady=(0, 25))
        header_frame.pack_propagate(False)
        
        # Title container for better positioning
        title_container = tk.Frame(header_frame, bg="#2c3e50")
        title_container.place(relx=0.5, rely=0.5, anchor="center")
        
        title_label = tk.Label(
            title_container,
            text="🎙️ Language Recognition",
            font=("Segoe UI", 24, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_container,
            text="Real-time Audio Language Detection",
            font=("Segoe UI", 10),
            bg="#2c3e50",
            fg="#95a5a6"
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Status section with modern card design
        status_card = tk.Frame(main_frame, bg="white", relief=tk.FLAT, bd=0)
        status_card.pack(fill=tk.X, pady=(0, 25))
        
        status_inner = tk.Frame(status_card, bg="white", padx=20, pady=15)
        status_inner.pack(fill=tk.X)
        
        tk.Label(
            status_inner, 
            text="Status", 
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#7f8c8d"
        ).pack(side=tk.LEFT)
        
        self.status_label = tk.Label(
            status_inner,
            text="● Ready",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#95a5a6"
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Modern control buttons with rounded appearance
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Button styling
        btn_style = {
            'font': ('Segoe UI', 12, 'bold'),
            'relief': tk.FLAT,
            'bd': 0,
            'cursor': 'hand2',
            'height': 2,
            'highlightthickness': 0
        }
        
        self.start_button = tk.Button(
            button_frame,
            text="▶  Start Recording",
            command=self.start_recording,
            bg=self.success_color,
            fg="white",
            activebackground="#229954",
            **btn_style
        )
        self.start_button.pack(fill=tk.X, pady=(0, 12))
        
        self.stop_button = tk.Button(
            button_frame,
            text="⏹  Stop Recording",
            command=self.stop_recording,
            bg=self.danger_color,
            fg="white",
            activebackground="#c0392b",
            state=tk.DISABLED,
            **btn_style
        )
        self.stop_button.pack(fill=tk.X, pady=(0, 12))
        
        self.file_button = tk.Button(
            button_frame,
            text="📂  Select Audio File",
            command=self.select_audio_file,
            bg=self.primary_color,
            fg="white",
            activebackground="#2980b9",
            **btn_style
        )
        self.file_button.pack(fill=tk.X)
        
        # Disable buttons if no recognizer is available
        if not self.whisper_recognizer and not self.groq_recognizer:
            self.start_button.config(state=tk.DISABLED)
            self.file_button.config(state=tk.DISABLED)
        
        # Modern detection card with shadow effect
        detection_card = tk.Frame(main_frame, bg="white", relief=tk.FLAT, bd=0)
        detection_card.pack(fill=tk.BOTH, pady=(0, 25))
        
        detection_inner = tk.Frame(detection_card, bg="white", padx=30, pady=25)
        detection_inner.pack(fill=tk.BOTH)
        
        # Card title
        tk.Label(
            detection_inner,
            text="Detected Language",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#7f8c8d"
        ).pack(anchor=tk.W, pady=(0, 15))
        
        # Language display with modern font
        self.language_label = tk.Label(
            detection_inner,
            text="---",
            font=("Segoe UI", 36, "bold"),
            bg="white",
            fg=self.primary_color
        )
        self.language_label.pack(pady=(0, 20))
        
        # Confidence section
        conf_label = tk.Label(
            detection_inner,
            text="Confidence Level",
            font=("Segoe UI", 10),
            bg="white",
            fg="#7f8c8d"
        )
        conf_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Modern progress bar
        self.confidence_canvas = tk.Canvas(
            detection_inner,
            height=40,
            bg="#ecf0f1",
            highlightthickness=0
        )
        self.confidence_canvas.pack(fill=tk.X, pady=(0, 8))
        
        self.confidence_text = tk.Label(
            detection_inner,
            text="0%",
            font=("Segoe UI", 13, "bold"),
            bg="white",
            fg="#34495e"
        )
        self.confidence_text.pack(anchor=tk.E)
        
        # Modern history card
        history_card = tk.Frame(main_frame, bg="white", relief=tk.FLAT, bd=0)
        history_card.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        history_inner = tk.Frame(history_card, bg="white", padx=20, pady=20)
        history_inner.pack(fill=tk.BOTH, expand=True)
        
        # Card title
        tk.Label(
            history_inner,
            text="Recent Detections",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#7f8c8d"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Listbox with modern styling
        list_frame = tk.Frame(history_inner, bg="white")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame, width=12)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(
            list_frame,
            font=("Consolas", 10),
            height=5,
            bg="#f8f9fa",
            fg="#2c3e50",
            selectbackground=self.primary_color,
            selectforeground="white",
            yscrollcommand=scrollbar.set,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0
        )
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
        # Modern export button
        self.export_button = tk.Button(
            main_frame,
            text="💾  Export History",
            command=self.export_history,
            font=("Segoe UI", 11, "bold"),
            bg=self.warning_color,
            fg="white",
            activebackground="#e67e22",
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            height=2,
            highlightthickness=0
        )
        self.export_button.pack(fill=tk.X)
    
    def start_recording(self):
        """Handler for Start Recording button."""
        if not self.whisper_recognizer and not self.groq_recognizer:
            messagebox.showerror(
                "Error",
                "No language recognizer is available. Please check initialization."
            )
            return
        
        try:
            # Clear session history for new recording session
            self.session_history.clear()
            self.history_listbox.delete(0, tk.END)
            
            # Reset current detection display
            self.language_label.config(text="---")
            self.confidence_canvas.delete("all")
            self.confidence_text.config(text="0%")
            
            # Initialize audio handler with callback
            chunk_duration = self.config.get('whisper', {}).get('chunk_duration', 3.0)
            sample_rate = self.config.get('whisper', {}).get('sample_rate', 16000)
            
            self.audio_handler = AudioStreamHandler(
                sample_rate=sample_rate,
                chunk_duration=chunk_duration,
                callback=self.on_audio_chunk_ready
            )
            
            # Start recording
            self.audio_handler.start_recording()
            
            # Update UI state
            self.status_label.config(text="● Recording", fg=self.success_color)
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.file_button.config(state=tk.DISABLED)
            
        except MicrophoneError as e:
            messagebox.showerror(
                "Microphone Error",
                f"Failed to access microphone:\n{str(e)}\n\n"
                "Please check that your microphone is connected and accessible."
            )
            self.start_button.config(state=tk.DISABLED)
        except AudioStreamError as e:
            messagebox.showerror(
                "Audio Stream Error",
                f"Failed to start recording:\n{str(e)}"
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Unexpected error starting recording:\n{str(e)}"
            )
    
    def stop_recording(self):
        """Handler for Stop Recording button."""
        try:
            if self.audio_handler:
                self.audio_handler.stop_recording()
                self.audio_handler = None
            
            # Update UI state
            self.status_label.config(text="● Ready", fg="#95a5a6")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.file_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error stopping recording:\n{str(e)}"
            )
    
    def on_audio_chunk_ready(self, audio_data, sample_rate):
        """
        Callback method invoked when an audio chunk is ready for processing.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
        """
        if self.is_processing:
            return  # Skip if already processing
        
        self.is_processing = True
        
        # Update status to Processing
        self.root.after(0, lambda: self.status_label.config(text="● Processing", fg=self.warning_color))
        
        # Process in separate thread to avoid blocking GUI
        def process_audio():
            try:
                # Get thresholds from config
                whisper_config = self.config.get('whisper', {})
                silence_threshold = whisper_config.get('silence_threshold', 0.005)
                min_confidence = whisper_config.get('min_confidence', 0.25)
                
                # Choose recognizer (Groq preferred if available)
                if self.use_groq and self.groq_recognizer:
                    recognizer = self.groq_recognizer
                else:
                    recognizer = self.whisper_recognizer
                
                # Recognize language from audio
                results = recognizer.recognize_from_audio(
                    audio_data, 
                    sample_rate,
                    silence_threshold=silence_threshold,
                    min_confidence=min_confidence
                )
                
                if results:
                    # Get top result (only one language returned now)
                    language, probability = results[0]
                    
                    # Create detection result
                    detection = DetectionResult(
                        language=language,
                        probability=probability,
                        timestamp=datetime.now(),
                        audio_duration=len(audio_data) / sample_rate,
                        source="microphone"
                    )
                    
                    # Add to session history
                    self.session_history.add_detection(detection)
                    
                    # Update GUI (must be done in main thread)
                    self.root.after(0, lambda: self.update_results(language, probability))
                    self.root.after(0, lambda: self.add_to_history(detection))
                else:
                    # No speech detected or confidence too low
                    self.root.after(0, lambda: self.clear_results())
                
            except (WhisperRecognizerError, GroqRecognizerError) as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Recognition Error",
                    f"Failed to recognize language:\n{str(e)}"
                ))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"Unexpected error during processing:\n{str(e)}"
                ))
            finally:
                # Update status back to Recording
                self.is_processing = False
                if self.audio_handler and self.audio_handler.is_recording():
                    self.root.after(0, lambda: self.status_label.config(text="● Recording", fg=self.success_color))
        
        # Start processing thread
        processing_thread = threading.Thread(target=process_audio, daemon=True)
        processing_thread.start()
    
    def select_audio_file(self):
        """Handler for Select Audio File button."""
        if not self.whisper_recognizer and not self.groq_recognizer:
            messagebox.showerror(
                "Error",
                "No language recognizer is available. Please check initialization."
            )
            return
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac"),
                ("WAV Files", "*.wav"),
                ("MP3 Files", "*.mp3"),
                ("FLAC Files", "*.flac"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return  # User cancelled
        
        # Show processing indicator
        self.status_label.config(text="● Processing", fg=self.warning_color)
        self.file_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        
        # Process in separate thread
        def process_file():
            try:
                # Get thresholds from config
                whisper_config = self.config.get('whisper', {})
                silence_threshold = whisper_config.get('silence_threshold', 0.005)
                min_confidence = whisper_config.get('min_confidence', 0.25)
                
                # Choose recognizer (Groq preferred if available)
                if self.use_groq and self.groq_recognizer:
                    recognizer = self.groq_recognizer
                else:
                    recognizer = self.whisper_recognizer
                
                # Recognize language from file
                results = recognizer.recognize_from_file(
                    file_path,
                    silence_threshold=silence_threshold,
                    min_confidence=min_confidence
                )
                
                if results:
                    # Only one result now
                    language, probability = results[0]
                    
                    result_text = f"Detected Language:\n\n{language.capitalize()}: {probability * 100:.2f}%"
                    
                    self.root.after(0, lambda: messagebox.showinfo(
                        "File Recognition Results",
                        result_text
                    ))
                    
                    # Create detection result
                    detection = DetectionResult(
                        language=language,
                        probability=probability,
                        timestamp=datetime.now(),
                        audio_duration=0.0,  # Unknown for file
                        source="file"
                    )
                    
                    # Add to session history
                    self.session_history.add_detection(detection)
                    
                    # Update GUI
                    self.root.after(0, lambda: self.update_results(language, probability))
                    self.root.after(0, lambda: self.add_to_history(detection))
                else:
                    # No speech detected or confidence too low
                    self.root.after(0, lambda: messagebox.showinfo(
                        "File Recognition Results",
                        "No speech detected or confidence too low.\n\nThe audio may be silent or not contain clear speech in supported languages."
                    ))
                    self.root.after(0, lambda: self.clear_results())
                
            except (WhisperRecognizerError, GroqRecognizerError) as e:
                error_msg = f"Failed to recognize language from file:\n{str(e)}"
                self.root.after(0, lambda msg=error_msg: messagebox.showerror(
                    "Recognition Error",
                    msg
                ))
            except Exception as e:
                error_msg = f"Error processing audio file:\n{str(e)}"
                self.root.after(0, lambda msg=error_msg: messagebox.showerror(
                    "File Error",
                    msg
                ))
            finally:
                # Restore UI state
                self.root.after(0, lambda: self.status_label.config(text="● Ready", fg="#95a5a6"))
                self.root.after(0, lambda: self.file_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
        
        # Start processing thread
        processing_thread = threading.Thread(target=process_file, daemon=True)
        processing_thread.start()
    
    def update_results(self, language: str, probability: float):
        """
        Update the current detection display with language and confidence.
        
        Args:
            language: Detected language name
            probability: Confidence probability (0.0 to 1.0)
        """
        # Update language label
        self.language_label.config(text=language.upper())
        
        # Update confidence bar
        self.confidence_canvas.delete("all")
        
        canvas_width = self.confidence_canvas.winfo_width()
        if canvas_width <= 1:  # Canvas not yet rendered
            canvas_width = 500  # Default width
        
        canvas_height = 30
        
        # Draw modern confidence bar with rounded corners effect
        bar_width = int(canvas_width * probability)
        
        # Color based on confidence level
        if probability >= 0.8:
            bar_color = self.success_color
        elif probability >= 0.6:
            bar_color = self.warning_color
        else:
            bar_color = self.danger_color
        
        # Draw filled portion
        if bar_width > 0:
            self.confidence_canvas.create_rectangle(
                0, 0, bar_width, canvas_height,
                fill=bar_color,
                outline=""
            )
        
        # Update confidence text
        self.confidence_text.config(text=f"{int(probability * 100)}%")
    
    def clear_results(self):
        """
        Clear the current detection display when no speech is detected.
        """
        # Update language label to show no speech
        self.language_label.config(text="NO SPEECH")
        
        # Clear confidence bar
        self.confidence_canvas.delete("all")
        
        canvas_width = self.confidence_canvas.winfo_width()
        if canvas_width <= 1:
            canvas_width = 500
        
        canvas_height = 30
        
        # Draw empty background
        self.confidence_canvas.create_rectangle(
            0, 0, canvas_width, canvas_height,
            fill="lightgray",
            outline=""
        )
        
        # Update confidence text
        self.confidence_text.config(text="0%")
    
    def add_to_history(self, detection: DetectionResult):
        """
        Add a detection result to the history listbox.
        
        Args:
            detection: DetectionResult object to add
        """
        # Format entry as "[HH:MM:SS] Language - XX%"
        time_str = detection.timestamp.strftime("%H:%M:%S")
        prob_percent = int(detection.probability * 100)
        entry = f"[{time_str}] {detection.language.capitalize()} - {prob_percent}%"
        
        # Add to listbox
        self.history_listbox.insert(tk.END, entry)
        
        # Keep only last 3 entries visible
        if self.history_listbox.size() > 3:
            self.history_listbox.delete(0)
        
        # Scroll to bottom
        self.history_listbox.see(tk.END)
    
    def export_history(self):
        """Handler for Export History button."""
        # Check if there's any history to export
        if not self.session_history.detections:
            messagebox.showinfo(
                "No History",
                "There is no detection history to export."
            )
            return
        
        # Open file save dialog
        file_path = filedialog.asksaveasfilename(
            title="Export History",
            defaultextension=".txt",
            filetypes=[
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ],
            initialfile=f"language_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Export history to file
            self.session_history.export_to_file(file_path)
            
            # Show success message
            messagebox.showinfo(
                "Export Successful",
                f"History exported successfully to:\n{file_path}"
            )
            
        except IOError as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export history:\n{str(e)}"
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Unexpected error during export:\n{str(e)}"
            )
    
    def _on_closing(self):
        """Handle window close event with proper cleanup."""
        try:
            # Stop recording if active
            if self.audio_handler and self.audio_handler.is_recording():
                self.audio_handler.stop_recording()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.root.destroy()
    
    def run(self):
        """Start the GUI application main loop."""
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()
