import asyncio
import websockets
import pyaudio
import webrtcvad
import time
import threading
import argparse
import queue
import sys
import os
import contextlib

import subprocess
import shutil

# Check for wtype availability
WTYPE_AVAILABLE = shutil.which('wtype') is not None
TYPING_AVAILABLE = WTYPE_AVAILABLE

# --- Configuration ---
WEBSOCKET_URI_BASE = "ws://localhost:5000/ws/transcribe_v3"
RATE = 16000  # Sample rate (16kHz)
VAD_CHUNK_DURATION_MS = 30 # VAD supports 10, 20, or 30 ms chunks
VAD_CHUNK_SIZE = int(RATE * VAD_CHUNK_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Mono audio
VAD_AGGRESSIVENESS = 3  # VAD aggressiveness (0-3)
SILENCE_TIMEOUT = 5  # Seconds of silence to wait before stopping

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output at file descriptor level"""
    # Save original stderr file descriptor
    stderr_fd = sys.stderr.fileno()
    old_stderr_fd = os.dup(stderr_fd)
    
    # Redirect stderr to /dev/null
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, stderr_fd)
    
    try:
        yield
    finally:
        # Restore original stderr
        os.dup2(old_stderr_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(old_stderr_fd)

class AudioClient:
    def __init__(self, send_interval_ms=250, verbose=False, script_start_time=None, type_mode=False, type_delay=None):
        # Suppress ALSA/JACK errors unless verbose mode
        if verbose:
            self.p = pyaudio.PyAudio()
        else:
            with suppress_stderr():
                self.p = pyaudio.PyAudio()
            
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.send_interval_ms = send_interval_ms
        self.send_interval_bytes = int(RATE * CHANNELS * (self.send_interval_ms / 1000.0))
        self.websocket_uri = f"{WEBSOCKET_URI_BASE}?chunk_duration_ms={self.send_interval_ms}"
        self.websocket = None
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.verbose = verbose
        self.script_start_time = script_start_time
        self.type_mode = type_mode
        self.type_delay = type_delay

    def _record_thread(self):
        stream = self.p.open(format=FORMAT,
                             channels=CHANNELS,
                             rate=RATE,
                             input=True,
                             frames_per_buffer=VAD_CHUNK_SIZE)
        
        silence_start_time = None
        buffer = bytearray()

        while not self.stop_event.is_set():
            data = stream.read(VAD_CHUNK_SIZE)
            is_speech = self.vad.is_speech(data, RATE)

            if is_speech:
                silence_start_time = None
                buffer.extend(data)
            else:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_TIMEOUT:
                    if self.verbose:
                        print("Silence detected. Stopping recording.", file=sys.stderr)
                    self.stop_event.set()
                # Still buffer non-speech to avoid cutting words off
                buffer.extend(data)

            while len(buffer) >= self.send_interval_bytes:
                self.audio_queue.put(buffer[:self.send_interval_bytes])
                buffer = buffer[self.send_interval_bytes:]

        stream.stop_stream()
        stream.close()

    async def receiver(self):
        try:
            while True:
                message = await self.websocket.recv()
                
                if self.type_mode and TYPING_AVAILABLE:
                    # Type using wtype - add space after message
                    text_to_type = message + ' '
                    try:
                        subprocess.run(['wtype', '-d', str(self.type_delay), text_to_type], check=True)
                    except subprocess.CalledProcessError as e:
                        if self.verbose:
                            print(f"wtype failed: {e}", file=sys.stderr)
                
                if self.verbose:
                    print(f"[{time.strftime('%H:%M:%S')}] Transcription: {message}", end=" ", flush=True, file=sys.stderr)
                    if not self.type_mode:  # Only print to stdout if not typing
                        print(message, end=" ", flush=True, file=sys.stdout)
                else:
                    if not self.type_mode:  # Only print to stdout if not typing
                        print(message, end=" ", flush=True, file=sys.stdout)
                        
        except websockets.exceptions.ConnectionClosed:
            print("\nConnection to server closed.", file=sys.stderr)
        except Exception as e:
            print(f"\nAn error occurred in receiver: {e}", file=sys.stderr)

    async def record_and_stream(self):
        recording_start_time = time.time()
        
        try:
            websocket_start_time = time.time()
            self.websocket = await websockets.connect(self.websocket_uri)
            
            if self.verbose:
                websocket_connect_time = time.time()
                print(f"[PROFILE] WebSocket connection: {websocket_connect_time - websocket_start_time:.3f}s", file=sys.stderr)
                if self.script_start_time:
                    print(f"[PROFILE] Total startup time: {websocket_connect_time - self.script_start_time:.3f}s", file=sys.stderr)
                print("Recording started...", file=sys.stderr)
            
            # Start the receiver task
            receiver_task = asyncio.create_task(self.receiver())

            # Start the recording thread
            record_thread = threading.Thread(target=self._record_thread)
            record_thread.start()

            while not self.stop_event.is_set():
                try:
                    data = self.audio_queue.get_nowait()
                    await self.websocket.send(data)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

            # Stop the tasks and thread
            record_thread.join()
            receiver_task.cancel()

        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)
        finally:
            if self.websocket:
                await self.websocket.close()
            self.p.terminate()
            if self.verbose:
                print("Recording stopped.", file=sys.stderr)

if __name__ == "__main__":
    script_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Client for streaming audio to the Parakeet ASR server.")
    parser.add_argument("--send_interval_ms", type=int, default=250,
                        help="Duration of each audio chunk sent to the server in milliseconds.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose printing with timestamps.")
    parser.add_argument("--type", type=int, nargs="?", const=10, help="Type transcription into the focused window instead of printing to stdout. Optional delay in milliseconds (default: 10).")
    args = parser.parse_args()
    
    if args.type and not TYPING_AVAILABLE:
        print("Error: --type mode requires wtype. Install it with your package manager.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        init_start_time = time.time()
        print(f"[PROFILE] Script startup: {init_start_time - script_start_time:.3f}s", file=sys.stderr)
    
    client = AudioClient(send_interval_ms=args.send_interval_ms, verbose=args.verbose, script_start_time=script_start_time, type_mode=args.type, type_delay=args.type if args.type else None)
    
    if args.verbose:
        client_init_time = time.time()
        print(f"[PROFILE] AudioClient initialization: {client_init_time - init_start_time:.3f}s", file=sys.stderr)
    
    asyncio.run(client.record_and_stream())
