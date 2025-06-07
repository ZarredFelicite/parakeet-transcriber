import asyncio
import websockets
import pyaudio
import webrtcvad
import time
import threading
import argparse
import queue

# --- Configuration ---
WEBSOCKET_URI = "ws://localhost:5000/ws/transcribe_v2"
CHUNK_DURATION_MS = 30  # VAD supports 10, 20, or 30 ms chunks
RATE = 16000  # Sample rate (16kHz)
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # 30ms chunk size in bytes
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Mono audio
VAD_AGGRESSIVENESS = 3  # VAD aggressiveness (0-3)
SILENCE_TIMEOUT = 5  # Seconds of silence to wait before stopping

class AudioClient:
    def __init__(self, verbose=False):
        self.p = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.websocket = None
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.verbose = verbose

    def _record_thread(self):
        stream = self.p.open(format=FORMAT,
                             channels=CHANNELS,
                             rate=RATE,
                             input=True,
                             frames_per_buffer=CHUNK_SIZE)
        
        silence_start_time = None
        buffer = bytearray()
        two_fifty_ms_bytes = RATE * CHANNELS // 2 # 0.25 second of 16-bit audio

        while not self.stop_event.is_set():
            data = stream.read(CHUNK_SIZE)
            buffer.extend(data)

            is_speech = self.vad.is_speech(data, RATE)

            if is_speech:
                silence_start_time = None
            else:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_TIMEOUT:
                    print("Silence detected. Stopping recording.")
                    self.stop_event.set()

            if len(buffer) >= two_fifty_ms_bytes:
                self.audio_queue.put(buffer)
                buffer = bytearray()

        stream.stop_stream()
        stream.close()

    async def receiver(self):
        try:
            while True:
                message = await self.websocket.recv()
                if self.verbose:
                    print(f"[{time.strftime('%H:%M:%S')}] Transcription: {message}")
                else:
                    print(f"Transcription: {message}")
        except websockets.exceptions.ConnectionClosed:
            print("Connection to server closed.")
        except Exception as e:
            print(f"An error occurred in receiver: {e}")

    async def record_and_stream(self):
        print("Recording started...")

        try:
            self.websocket = await websockets.connect(WEBSOCKET_URI)
            
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
            print(f"An error occurred: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()
            self.p.terminate()
            print("Recording stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for streaming audio to the Parakeet ASR server.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose printing with timestamps.")
    args = parser.parse_args()

    client = AudioClient(verbose=args.verbose)
    asyncio.run(client.record_and_stream())
