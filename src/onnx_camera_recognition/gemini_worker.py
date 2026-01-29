import asyncio
import sys
import os
import pyaudio
from google import genai
 
# --- Configuration ---
# Please enter your API Key here
API_KEY = "" 
 
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
 
# PyAudio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECEIVE_SAMPLE_RATE = 24000

 # Global queue
audio_queue_output = asyncio.Queue()
 
# Global state
is_playing = False

async def handle_stdin(session):
    """
    Read standard input (stdin) in a separate thread
    When text arrives from the main program, send it to Gemini
    """
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        
        text_input = line.decode().strip()
        if text_input:
            # Send to Gemini
            # print(f"Sending to API: {text_input}", file=sys.stderr) # Send debug info to stderr to avoid interfering with the stdout protocol
            await session.send_realtime_input(text=text_input)

async def receive_audio(session):
    """Receive Gemini's response and enqueue for playback"""
    while True:
        try:
            turn = session.receive()
            async for response in turn:
                if response.server_content and response.server_content.model_turn:
                    for part in response.server_content.model_turn.parts:
                        if part.inline_data and isinstance(part.inline_data.data, bytes):
                            audio_queue_output.put_nowait(part.inline_data.data)
        except Exception as e:
            print(f"Error receiving audio: {e}", file=sys.stderr)
            break

async def play_audio():
    """Dequeue audio and play, while reporting status to the main program"""
    global is_playing
    
    pya = pyaudio.PyAudio()
    stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
    )

    while True:
        # Wait for audio data
        bytestream = await audio_queue_output.get()
        
        # State switching logic: if not previously playing and now starting -> send SPEAKING signal
        if not is_playing:
            is_playing = True
            print("STATUS:SPEAKING")
            sys.stdout.flush()

        # Play audio (run blocking operation in a thread)
        await asyncio.to_thread(stream.write, bytestream)

        # Check whether the queue is empty
        if audio_queue_output.empty():
            # Wait briefly to avoid short dropouts caused by network jitter
            await asyncio.sleep(0.2)
            if audio_queue_output.empty():
                is_playing = False
                print("STATUS:SILENT")
                sys.stdout.flush()

async def run():
    client = genai.Client(api_key=API_KEY)
    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": "You are a helpful and friendly AI assistant.",
    }

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("Gemini Connected.", file=sys.stderr)
            
            async with asyncio.TaskGroup() as tg:
                tg.create_task(handle_stdin(session))
                tg.create_task(receive_audio(session))
                tg.create_task(play_audio())

    except Exception as e:
        print(f"Connection Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
