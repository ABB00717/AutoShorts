import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
from tqdm import tqdm

def process_video_chunk(video_path, start_time, end_time, output_chunk_path):
    """
    Process a chunk of video between start_time and end_time.
    
    Args:
        video_path (str): Path to the input video file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        output_chunk_path (str): Path to save the processed chunk
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Load the video chunk
        video = VideoFileClip(video_path).subclip(start_time, end_time)
        chunk_duration = video.duration
        
        # Save the chunk to a temporary file
        temp_chunk_path = f"temp_chunk_{start_time}_{end_time}.mp4"
        video.write_videofile(temp_chunk_path, codec="libx264", audio_codec="aac")
        video.close()
        
        # Transcribe the chunk
        model = whisper.load_model("base")
        result = model.transcribe(temp_chunk_path, 
                                prompt="Umm,let me think like,hmm... Okay,here's what I'm,like,thinking.",
                                word_timestamps=True)
        
        # Clean up temporary chunk file
        os.remove(temp_chunk_path)
        
        # Identify silence periods
        silence_periods = identify_silence_periods(result, chunk_duration, threshold=0.5)
        
        # Adjust silence periods to account for chunk start time
        adjusted_silence_periods = [(start + start_time, end + start_time) for start, end in silence_periods]
        
        # Cut silences from the chunk
        cut_silences(video_path, output_chunk_path, adjusted_silence_periods, start_time, end_time)
        
        return True
    except Exception as e:
        print(f"Error processing chunk from {start_time} to {end_time}: {str(e)}")
        if os.path.exists(temp_chunk_path):
            os.remove(temp_chunk_path)
        return False

def identify_silence_periods(transcription, video_duration, threshold=1.0, buffer=0.1):
    """
    Identifies silence periods in the transcription based on the threshold.
    
    Args:
        transcription (dict): The transcription result with word-level timestamps.
        threshold (float): The minimum duration of silence to be considered.
        video_duration (float): Duration of the video chunk.
    
    Returns:
        list: A list of tuples where each tuple contains the start and end time of a silence period.
    """
    silence_periods = []
    words = transcription['segments']
    previous_end = 0

    for word in words:
        start_time = word['start']
        if start_time - previous_end > threshold:
            # Ensure we don't exceed the chunk duration
            end_time = min(start_time - buffer, video_duration)
            if end_time > previous_end + buffer:
                silence_periods.append((previous_end + buffer, end_time))
        previous_end = word['end']

    # Handle the final silence period
    if video_duration - previous_end > threshold:
        end_time = min(video_duration - buffer, video_duration)
        if end_time > previous_end + buffer:
            silence_periods.append((previous_end + buffer, end_time))

    return silence_periods

def cut_silences(input_video, output_video, silence_periods, start_time=None, end_time=None):
    """
    Removes the silence periods from the video and saves the result.
    
    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to save the output video file.
        silence_periods (list): A list of tuples indicating silence periods (start, end).
        start_time (float): Optional start time for the video chunk
        end_time (float): Optional end time for the video chunk
    """
    try:
        # Load the video
        video = VideoFileClip(input_video)
        if start_time is not None and end_time is not None:
            video = video.subclip(start_time, end_time)

        # Create a list of clips without the silence periods
        clips = []
        last_end = 0

        for (start, end) in silence_periods:
            # Ensure we don't exceed the video duration
            if start > video.duration:
                break
            if last_end < start:
                clips.append(video.subclip(last_end, min(start, video.duration)))
            last_end = min(end, video.duration)

        # Add the final clip if there's any remaining video after the last silence
        if last_end < video.duration:
            clips.append(video.subclip(last_end, video.duration))

        # Concatenate the remaining clips
        if clips:
            final_clip = concatenate_videoclips(clips)
            # Write the result to a file
            final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")
            final_clip.close()
        else:
            # If no clips are left after cutting silences, save the original video
            video.write_videofile(output_video, codec="libx264", audio_codec="aac")
        
        video.close()
    except Exception as e:
        print(f"Error in cut_silences: {str(e)}")
        if 'video' in locals():
            video.close()

def process_long_video(input_video, output_video, chunk_duration=300):
    """
    Process a long video by breaking it into chunks.
    
    Args:
        input_video (str): Path to the input video file
        output_video (str): Path to save the output video file
        chunk_duration (int): Duration of each chunk in seconds (default: 300 seconds = 5 minutes)
    """
    # Load the video to get its duration
    video = VideoFileClip(input_video)
    total_duration = video.duration
    video.close()
    
    # Calculate number of chunks
    num_chunks = int(total_duration / chunk_duration) + (1 if total_duration % chunk_duration > 0 else 0)
    
    # Create temporary directory for chunks
    temp_dir = "temp_chunks"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each chunk
    processed_chunks = []
    for i in tqdm(range(num_chunks), desc="Processing video chunks"):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, total_duration)
        
        # Skip if this chunk would be too short (less than 1 second)
        if end_time - start_time < 1:
            continue
            
        chunk_output = os.path.join(temp_dir, f"chunk_{i}.mp4")
        if process_video_chunk(input_video, start_time, end_time, chunk_output):
            processed_chunks.append(chunk_output)
    
    # Combine all processed chunks
    if processed_chunks:
        print("Combining processed chunks...")
        final_clips = []
        for chunk_path in processed_chunks:
            try:
                clip = VideoFileClip(chunk_path)
                if clip.duration > 0:  # Only add non-empty clips
                    final_clips.append(clip)
                else:
                    clip.close()
                    os.remove(chunk_path)
            except Exception as e:
                print(f"Error loading chunk {chunk_path}: {str(e)}")
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        if final_clips:
            print(f"Combining {len(final_clips)} chunks...")
            final_video = concatenate_videoclips(final_clips)
            print(f"Writing final video to {output_video}...")
            final_video.write_videofile(output_video, codec="libx264", audio_codec="aac")
            final_video.close()
            
            # Clean up
            for clip in final_clips:
                clip.close()
            for chunk in processed_chunks:
                if os.path.exists(chunk):
                    os.remove(chunk)
            os.rmdir(temp_dir)
            print("Processing completed successfully!")
        else:
            print("No valid clips to combine.")
    else:
        print("No chunks were successfully processed.")

if __name__ == "__main__":
    video_path = "input_video.mp4"       # Path to your video file
    output_path = "output_video.mp4"    # Path to save the edited video
    
    # Process the video in chunks
    process_long_video(video_path, output_path)
