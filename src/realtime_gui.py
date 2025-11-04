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
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Real-time Language Recognition")
        self.root.geometry("600x700")
        self.root.resizable(False, False)
        
        # Initialize components
        self.whisper_recognizer: Optional[WhisperLanguageRecognizer] = None
        self.audio_handler: Optional[AudioStreamHandler] = None
        self.session_history = SessionHistory()
        
        # State variables
        self.is_processing = False
        
        # Initialize Whisper recognizer
        self._initialize_whisper()
        
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
    
    def _initialize_whisper(self):
        """Initialize WhisperLanguageRecognizer with error handling."""
        try:
            model_size = self.config.get('whisper', {}).get('model_size', 'base')
            self.whisper_recognizer = WhisperLanguageRecognizer(model_size=model_size)
        except ModelLoadError as e:
            messagebox.showerror(
                "Model Load Error",
                f"Failed to load Whisper model.\n\n{str(e)}\n\n"
                "Please install openai-whisper:\npip install openai-whisper"
            )
            self.whisper_recognizer = None
        except Exception as e:
            messagebox.showerror(
                "Initialization Error",
                f"Failed to initialize Whisper recognizer:\n{str(e)}"
            )
            self.whisper_recognizer = None
    
    def _setup_gui(self):
        """Setup the GUI layout and all widgets."""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Real-time Language Recognition",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Status section
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(status_frame, text="Status:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.status_label = ttk.Label(
            status_frame,
            text="Idle",
            font=("Arial", 10),
            foreground="gray"
        )
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Control buttons section
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.start_button = ttk.Button(
            button_frame,
            text="Start Recording",
            command=self.start_recording,
            width=20
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop Recording",
            command=self.stop_recording,
            width=20,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_button = ttk.Button(
            button_frame,
            text="Select Audio File",
            command=self.select_audio_file,
            width=20
        )
        self.file_button.pack(side=tk.LEFT)
        
        # Disable buttons if Whisper is not available
        if not self.whisper_recognizer:
            self.start_button.config(state=tk.DISABLED)
            self.file_button.config(state=tk.DISABLED)
        
        # Current detection display section
        detection_frame = ttk.LabelFrame(
            main_frame,
            text="Current Detection",
            padding="15"
        )
        detection_frame.pack(fill=tk.BOTH, pady=(0, 20))
        
        # Language label
        lang_label_frame = ttk.Frame(detection_frame)
        lang_label_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(lang_label_frame, text="Language:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.language_label = ttk.Label(
            lang_label_frame,
            text="---",
            font=("Arial", 12, "bold"),
            foreground="blue"
        )
        self.language_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Confidence bar
        conf_label_frame = ttk.Frame(detection_frame)
        conf_label_frame.pack(fill=tk.X)
        
        ttk.Label(conf_label_frame, text="Confidence:", font=("Arial", 10)).pack(anchor=tk.W)
        
        # Canvas for confidence bar
        self.confidence_canvas = tk.Canvas(
            detection_frame,
            height=30,
            bg="white",
            highlightthickness=1,
            highlightbackground="gray"
        )
        self.confidence_canvas.pack(fill=tk.X, pady=(5, 0))
        
        self.confidence_text = ttk.Label(
            detection_frame,
            text="0%",
            font=("Arial", 9)
        )
        self.confidence_text.pack(anchor=tk.E, pady=(2, 0))
        
        # Detection history section
        history_frame = ttk.LabelFrame(
            main_frame,
            text="Detection History (Last 3)",
            padding="15"
        )
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(history_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(
            list_frame,
            font=("Courier", 10),
            height=3,
            yscrollcommand=scrollbar.set
        )
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
        # Export button
        self.export_button = ttk.Button(
            main_frame,
            text="Export History",
            command=self.export_history,
            width=20
        )
        self.export_button.pack(pady=(0, 10))
    
    def start_recording(self):
        """Handler for Start Recording button."""
        if not self.whisper_recognizer:
            messagebox.showerror(
                "Error",
                "Whisper recognizer is not available. Please check initialization."
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
            self.status_label.config(text="Recording", foreground="green")
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
            self.status_label.config(text="Idle", foreground="gray")
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
        self.root.after(0, lambda: self.status_label.config(text="Processing", foreground="orange"))
        
        # Process in separate thread to avoid blocking GUI
        def process_audio():
            try:
                # Get thresholds from config
                whisper_config = self.config.get('whisper', {})
                silence_threshold = whisper_config.get('silence_threshold', 0.005)
                min_confidence = whisper_config.get('min_confidence', 0.25)
                
                # Recognize language from audio
                results = self.whisper_recognizer.recognize_from_audio(
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
                
            except WhisperRecognizerError as e:
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
                    self.root.after(0, lambda: self.status_label.config(text="Recording", foreground="green"))
        
        # Start processing thread
        processing_thread = threading.Thread(target=process_audio, daemon=True)
        processing_thread.start()
    
    def select_audio_file(self):
        """Handler for Select Audio File button."""
        if not self.whisper_recognizer:
            messagebox.showerror(
                "Error",
                "Whisper recognizer is not available. Please check initialization."
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
        self.status_label.config(text="Processing File", foreground="orange")
        self.file_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        
        # Process in separate thread
        def process_file():
            try:
                # Get thresholds from config
                whisper_config = self.config.get('whisper', {})
                silence_threshold = whisper_config.get('silence_threshold', 0.005)
                min_confidence = whisper_config.get('min_confidence', 0.25)
                
                # Recognize language from file
                results = self.whisper_recognizer.recognize_from_file(
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
                
            except WhisperRecognizerError as e:
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
                self.root.after(0, lambda: self.status_label.config(text="Idle", foreground="gray"))
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
        
        # Draw background
        self.confidence_canvas.create_rectangle(
            0, 0, canvas_width, canvas_height,
            fill="lightgray",
            outline=""
        )
        
        # Draw confidence bar
        bar_width = int(canvas_width * probability)
        
        # Color based on confidence level
        if probability >= 0.8:
            bar_color = "green"
        elif probability >= 0.6:
            bar_color = "orange"
        else:
            bar_color = "red"
        
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
