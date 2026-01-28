#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import subprocess
import threading
import sys
import json
import os
import vosk
import queue
import time
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData

# --- Configuration ---
SLAVE_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_worker.py")
# Ensure this path is correct
PYTHON_INTERPRETER = "/home/mustar/miniconda3/envs/yolov11/bin/python" 
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
SILENCE_TIMEOUT = 2

class SpeechInteractionNode:
    def __init__(self):
        rospy.init_node('voice_interaction_node', anonymous=True)

        # 1. Initialize Vosk
        if not os.path.exists(VOSK_MODEL_PATH):
            rospy.logerr(f"Vosk model not found at {VOSK_MODEL_PATH}")
            sys.exit(1)
        self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
        self.rec = vosk.KaldiRecognizer(self.vosk_model, 16000)
        
        # 2. State flags
        self.ai_is_speaking = False
        self.expression_received = False
        self.accumulated_text = ""
        self.last_speech_time = time.time()
        self.shutdown_requested = False  # Added: shutdown request flag

        # 3. Start Gemini Worker (grandchild process)
        rospy.loginfo("Launching Gemini Worker...")
        self.gemini_process = subprocess.Popen(
            [PYTHON_INTERPRETER, "-u", SLAVE_SCRIPT_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # 4. Thread management
        threading.Thread(target=self.monitor_gemini_output, daemon=True).start()
        threading.Thread(target=self.monitor_gemini_error, daemon=True).start()
        threading.Thread(target=self.monitor_parent_input, daemon=True).start()

        # 5. ROS subscribe to microphone
        self.sub_mic = rospy.Subscriber("/mic/audio", AudioData, self.audio_callback)

        rospy.loginfo("Speech Node Ready. Waiting for expression from Master (ONNX)...")

    def monitor_parent_input(self):
        """Listen to commands from the master (ONNX) via stdin"""
        while not rospy.is_shutdown():
            try:
                line = sys.stdin.readline()
                if not line:
                    break 
                
                cmd = line.strip()
                if cmd == "CMD:SHUTDOWN":
                    rospy.loginfo("Received SHUTDOWN command from Master.")
                    self.perform_graceful_shutdown()
                    break
                elif cmd:
                    rospy.loginfo(f"Received expression from Master: {cmd}")
                    self.handle_expression_trigger(cmd)
            except Exception as e:
                rospy.logerr(f"Error reading from parent: {e}")
                break

    def perform_graceful_shutdown(self):
        """Perform graceful shutdown sequence"""
        self.shutdown_requested = True
        
        # 1. Stop receiving microphone data immediately
        if self.sub_mic:
            self.sub_mic.unregister()
            self.sub_mic = None
            rospy.loginfo("Microphone input stopped.")

        # 2. Wait for AI to finish speaking (check ai_is_speaking)
        # Give a max wait time (e.g., 30s) to avoid deadlock
        wait_start = time.time()
        while self.ai_is_speaking:
            if time.time() - wait_start > 30:
                rospy.logwarn("Timeout waiting for AI to finish speaking. Forcing shutdown.")
                break
            rospy.loginfo_throttle(1, "Waiting for AI to finish speaking...")
            time.sleep(0.5)

        rospy.loginfo("AI finished speaking or silent. Shutting down processes.")

        # 3. Terminate Gemini process
        if self.gemini_process:
            self.gemini_process.terminate()
            try:
                self.gemini_process.wait(timeout=2)
            except:
                self.gemini_process.kill()

        # 4. Exit this ROS node
        rospy.signal_shutdown("Graceful shutdown completed")
        sys.exit(0)

    def handle_expression_trigger(self, expression):
        if self.shutdown_requested: return # Ignore new triggers during shutdown

        if not self.expression_received:
            prompt = f"Assuming you can see my expression {expression}, I will share it with you."
            self.send_to_gemini(prompt)
            self.expression_received = True
            rospy.loginfo(f"Expression Context Sent to Gemini: {prompt}")

    def monitor_gemini_output(self):
        while not rospy.is_shutdown():
            if self.gemini_process.poll() is not None:
                rospy.logerr("Gemini process died!")
                break
            
            line = self.gemini_process.stdout.readline()
            if not line: continue
            
            line = line.strip()
            if line == "STATUS:SPEAKING":
                self.ai_is_speaking = True
            elif line == "STATUS:SILENT":
                self.ai_is_speaking = False
            else:
                print(f"[Gemini]: {line}") 

    def monitor_gemini_error(self):
        for line in self.gemini_process.stderr:
            rospy.logerr(f"[Gemini Error]: {line.strip()}")

    def send_to_gemini(self, text):
        if self.shutdown_requested: return # Do not send during shutdown

        if self.gemini_process.poll() is None:
            try:
                self.gemini_process.stdin.write(text + "\n")
                self.gemini_process.stdin.flush()
            except Exception as e:
                rospy.logerr(f"Failed to write to Gemini: {e}")

    def audio_callback(self, msg):
        # Ignore microphone if speaking or shutdown requested
        if self.ai_is_speaking or self.shutdown_requested:
            return

        data = msg.data
        if not isinstance(data, bytes):
            data = bytes(data)

        if len(data) == 0: return

        if self.rec.AcceptWaveform(data):
            result = json.loads(self.rec.Result())
            text = result.get("text", "")
            if text.strip():
                self.accumulated_text += " " + text
                self.last_speech_time = time.time()
                rospy.loginfo(f"Partial Full: {text}")
        else:
            partial = json.loads(self.rec.PartialResult())
            if partial.get("partial", "").strip():
                self.last_speech_time = time.time()

        current_time = time.time()
        if self.accumulated_text.strip() and (current_time - self.last_speech_time > SILENCE_TIMEOUT):
            final_text = self.accumulated_text.strip()
            rospy.loginfo(f"User Said: {final_text}")
            self.send_to_gemini(final_text)
            self.accumulated_text = ""

    def run(self):
        rospy.spin()
        if self.gemini_process:
            self.gemini_process.terminate()

if __name__ == '__main__':
    try:
        SpeechInteractionNode().run()
    except rospy.ROSInterruptException:
        pass