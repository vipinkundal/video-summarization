import gradio as gr
import torch
import yt_dlp
import os
import subprocess
import json
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM
import spaces
import moviepy.editor as mp
import time
import langdetect
import uuid

HF_TOKEN = os.environ.get("HF_TOKEN")
print("Starting the program...")

model_path = "Qwen/Qwen2.5-7B-Instruct"

torch.device('cpu')

#FOR GPU
# print(f"Loading model {model_path}...")
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
# model = model.eval()
# print("Model successfully loaded.")

#FOR CPU
print(f"Loading model {model_path}...")
# Configure model loading parameters
model_kwargs = {
    "trust_remote_code": True,
    "device_map": "auto",
    "torch_dtype": torch.float32,
    "low_cpu_mem_usage": True,
}
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
model = model.eval()

print("Model successfully loaded.")

model = model.eval()

def generate_unique_filename(extension):
    return f"{uuid.uuid4()}{extension}"

def cleanup_files(*files):
    for file in files:
        if file and os.path.exists(file):
            os.remove(file)
            print(f"Removed file: {file}")

def download_youtube_audio(url):
    print(f"Downloading audio from YouTube: {url}")
    output_path = generate_unique_filename(".wav")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': output_path,
        'keepvideo': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Check if the file was renamed to .wav.wav
    if os.path.exists(output_path + ".wav"):
        os.rename(output_path + ".wav", output_path)
    
    if os.path.exists(output_path):
        print(f"Audio download completed. File saved at: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    else:
        print(f"Error: File {output_path} not found after download.")
    
    return output_path

# @spaces.GPU(duration=90)
def transcribe_audio(file_path):
    print(f"Starting transcription of file: {file_path}")
    temp_audio = None
    if file_path.endswith(('.mp4', '.avi', '.mov', '.flv')):
        print("Video file detected. Extracting audio...")
        try:
            video = mp.VideoFileClip(file_path)
            temp_audio = generate_unique_filename(".wav")
            video.audio.write_audiofile(temp_audio)
            file_path = temp_audio
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            raise
    
    print(f"Does the file exist? {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'} bytes")
    
    output_file = generate_unique_filename(".json")
    command = [
        "insanely-fast-whisper",
        "--file-name", file_path,
        "--device", "cpu",
        # "--device-id", "0", #FOR GPU
        # "--model-name", "openai/whisper-large-v3",
        "--model-name", "openai/whisper-medium",
        "--task", "transcribe",
        "--timestamp", "chunk",
        "--transcript-path", output_file
    ]
    print(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Standard output: {result.stdout}")
        print(f"Error output: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running insanely-fast-whisper: {e}")
        print(f"Standard output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        raise
    
    print(f"Reading transcription file: {output_file}")
    try:
        with open(output_file, "r") as f:
            transcription = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"File content: {open(output_file, 'r').read()}")
        raise
    
    if "text" in transcription:
        result = transcription["text"]
    else:
        result = " ".join([chunk["text"] for chunk in transcription.get("chunks", [])])
    
    print("Transcription completed.")
    
    # Cleanup
    cleanup_files(output_file)
    if temp_audio:
        cleanup_files(temp_audio)
    
    return result

# @spaces.GPU(duration=90)
def generate_summary_stream(transcription):
    print("Starting summary generation...")
    print(f"Transcription length: {len(transcription)} characters")
    
    detected_language = langdetect.detect(transcription)
    
    prompt = f"""Summarize the following video transcription in 150-300 words. 
    The summary should be in the same language as the transcription, which is detected as {detected_language}.
    Please ensure that the summary captures the main points and key ideas of the transcription:

    {transcription[:300000]}..."""
    
    response, history = model.chat(tokenizer, prompt, history=[])
    print(f"Final summary generated: {response[:100]}...")
    print("Summary generation completed.")
    return response

def process_youtube(url):
    if not url:
        print("YouTube URL not provided.")
        return "Please enter a YouTube URL.", None
    print(f"Processing YouTube URL: {url}")
    
    audio_file = None
    try:
        audio_file = download_youtube_audio(url)
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"File {audio_file} does not exist after download.")
        
        print(f"Audio file found: {audio_file}")
        print("Starting transcription...")
        transcription = transcribe_audio(audio_file)
        print(f"Transcription completed. Length: {len(transcription)} characters")
        return transcription, None
    except Exception as e:
        print(f"Error processing YouTube: {e}")
        return f"Processing error: {str(e)}", None
    finally:
        if audio_file and os.path.exists(audio_file):
            cleanup_files(audio_file)
        print(f"Directory content after processing: {os.listdir('.')}")

def process_uploaded_video(video_path):
    print(f"Processing uploaded video: {video_path}")
    try:
        print("Starting transcription...")
        transcription = transcribe_audio(video_path)
        print(f"Transcription completed. Length: {len(transcription)} characters")
        return transcription, None
    except Exception as e:
        print(f"Error processing video: {e}")
        return f"Processing error: {str(e)}", None

print("Setting up Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎥 Video Transcription and Smart Summary
        
        Upload a video or provide a YouTube link to get a transcription and AI-generated summary. HF Zero GPU has a usage time limit. So if you want to run longer videos I recommend you clone the space. Remove @Spaces.gpu from the code and run it locally on your GPU!
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("📤 Video Upload"):
            video_input = gr.Video(label="Drag and drop or click to upload")
            video_button = gr.Button("🚀 Process Video", variant="primary")
        
        with gr.TabItem("🔗 YouTube Link"):
            url_input = gr.Textbox(label="Paste YouTube URL here", placeholder="https://www.youtube.com/watch?v=...")
            url_button = gr.Button("🚀 Process URL", variant="primary")
    
    with gr.Row():
        with gr.Column():
            transcription_output = gr.Textbox(label="📝 Transcription", lines=10, show_copy_button=True)
        with gr.Column():
            summary_output = gr.Textbox(label="📊 Summary", lines=10, show_copy_button=True)
    
    summary_button = gr.Button("📝 Generate Summary", variant="secondary")
    
    gr.Markdown(
        """
        ### How to use:
        1. Upload a video or paste a YouTube link.
        2. Click 'Process' to get the transcription.
        3. Click 'Generate Summary' to get a summary of the content.
        
        *Note: Processing may take a few minutes depending on the video length.*
        """
    )
    
    def process_video_and_update(video):
        if video is None:
            return "No video uploaded.", "Please upload a video."
        print(f"Video received: {video}")
        transcription, _ = process_uploaded_video(video)
        print(f"Returned transcription: {transcription[:100] if transcription else 'No transcription generated'}...")
        return transcription or "Transcription error", ""

    video_button.click(process_video_and_update, inputs=[video_input], outputs=[transcription_output, summary_output])
    url_button.click(process_youtube, inputs=[url_input], outputs=[transcription_output, summary_output])
    summary_button.click(generate_summary_stream, inputs=[transcription_output], outputs=[summary_output])


if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.queue()  # Enable queuing for better memory management
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
