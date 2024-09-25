import logging
import os
import subprocess
import re
from deepmultilingualpunctuation import PunctuationModel


def clean_subtitles(subtitle_text):
    subtitle_text = re.sub(r'(WEBVTT|Kind: captions|Language: \w{2}).*\n?', '', subtitle_text)
    # Step 1: Remove all timestamp lines (like "00:00:00.000 --> 00:00:03.139")
    cleaned_text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*\n', '', subtitle_text)

    # Step 2: Remove any remaining time-related tags (like "<00:00:00.870><c>")
    cleaned_text = re.sub(r'<.*?>', '', cleaned_text)

    # Step 3: Split lines and remove any exact duplicates
    lines = cleaned_text.splitlines()
    unique_lines = []
    last_line = None

    for line in lines:
        line = line.strip()
        # Only add the line if it's not a duplicate of the last one
        if line and line != last_line:
            unique_lines.append(line)
            last_line = line

    # Step 4: Join the lines into a single block of text
    joined_text = ' '.join(unique_lines)

    # Return the processed text with correct sentence boundaries and full stops
    return joined_text


class YouTubeTranscriptDownloader:
    def __init__(self, video_url, language='en'):
        self.video_url = video_url
        self.language = language

    def download_transcript(self, output_dir='./'):

        # Step 1: Check available subtitles
        available_subs = self.get_available_subtitles()

        if not available_subs:
            logging.info("No subtitles found for the video.")
            return

        # Try to download the selected subtitles (either manual or auto)
        try:
            # Step 2: Check for manual subtitles first
            if available_subs["manual"]:  # Check if manual subtitles are available
                logging.info(f"Downloading manual subtitles in {self.language}...")
                self._download_subtitles(output_dir, auto_generated=False)

            elif available_subs["auto"]:  # If no manual subs, check for auto-generated ones
                logging.info(
                    f"Manual subtitles not available in {self.language}. Downloading auto-generated subtitles...")
                self._download_subtitles(output_dir, auto_generated=True)

            else:
                logging.info(f"No subtitles available in {self.language}.")

            # Step 3: Check if the .srt file exists and return its content
            srt_file = self._find_srt_file(output_dir)
            if srt_file:
                srt_file_path = os.path.join(output_dir, srt_file)

                # Read the content of the subtitle file
                transcript = self._read_file(srt_file_path)

                # Remove the subtitle file after reading it
                try:
                    os.remove(srt_file_path)
                    logging.info(f"{srt_file_path} deleted successfully.")
                except OSError as e:
                    logging.error(f"Error while deleting subtitle file: {e}")

                return transcript

        except subprocess.CalledProcessError as e:
            raise Exception(f"Error occurred during subtitle download: {e}")

    def get_available_subtitles(self):
        """Check if manual or auto-generated subtitles are available for the video."""
        try:
            # Run the yt-dlp command to get the subtitles
            command = [
                'yt-dlp',
                '--list-subs',  # List available subtitles
                self.video_url  # Video URL
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True, encoding="utf-8")
            output = result.stdout

            # Initialize flags to hold whether auto or manual subtitles are found
            found_auto_subs = False
            found_manual_subs = False

            # Parse the output
            in_auto_subs_section = False
            in_manual_subs_section = False

            for line in output.splitlines():
                line = line.strip()

                # Identify when automatic and manual subtitles sections start
                if '[info] Available automatic captions' in line:
                    in_auto_subs_section = True
                    in_manual_subs_section = False
                elif '[info] Available subtitles' in line:
                    in_manual_subs_section = True
                    in_auto_subs_section = False
                elif line.startswith('[info]') or line.startswith('[youtube]'):
                    # End of sections
                    in_auto_subs_section = False
                    in_manual_subs_section = False
                elif in_auto_subs_section or in_manual_subs_section:
                    if not line.startswith('Language'):
                        # Split the line to extract language and name
                        parts = line.split()
                        lang_code = parts[0]  # First part is the language code
                        if in_auto_subs_section and lang_code == self.language:
                            found_auto_subs = True
                            break  # Break early if auto subs are found
                        elif in_manual_subs_section and lang_code == self.language:
                            found_manual_subs = True
                            break  # Break early if manual subs are found

            return {
                "manual": found_manual_subs,
                "auto": found_auto_subs
            }

        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred while listing subtitles: {e}")
            return None

    def _download_subtitles(self, output_dir, auto_generated):
        """Download manual or auto-generated subtitles."""
        if auto_generated:
            command = [
                'yt-dlp',
                '--write-auto-subs',  # Auto-generated subtitles
                '--skip-download',
                '--sub-lang', self.language,
                '--sub-format', 'srt',
                '--output', os.path.join(output_dir, '%(title)s.%(ext)s'),
                '--verbose',
                self.video_url  # Ensure video URL is included here
            ]
        else:
            command = [
                'yt-dlp',
                '--write-subs',  # Manual subtitles
                '--skip-download',
                '--sub-lang', self.language,
                '--sub-format', 'srt',
                '--output', os.path.join(output_dir, '%(title)s.%(ext)s'),
                '--verbose',
                self.video_url  # Ensure video URL is included here
            ]

        subprocess.run(command, check=True)

    def _find_srt_file(self, output_dir):
        """Find the .srt file in the output directory."""
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".vtt") or file_name.endswith('.srt'):
                return file_name
        return None

    def _read_file(self, file_path):
        """Read the content of the SRT file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()


def get_transcript(video_url, language='en'):
    # Create an instance of the downloader with the video URL and language
    transcript_downloader = YouTubeTranscriptDownloader(video_url, language=language)

    try:
        # Step 1: Download the transcript
        transcript_text = transcript_downloader.download_transcript()
        logging.info(transcript_text)

        # Step 2: Clean the downloaded subtitles
        clean_text = clean_subtitles(transcript_text)

        # Step 3: Restore punctuation using the PunctuationModel
        model = PunctuationModel()
        punctuated_text = model.restore_punctuation(clean_text)

        # Step 4: Return the result
        return punctuated_text
    except Exception as e:
        logging.error(f"Error: {e}")

# video_url = "https://www.youtube.com/watch?v=LIpNBNlBpbQ"
# video_url = "https://www.youtube.com/watch?v=VSrLbzZzJU8"
