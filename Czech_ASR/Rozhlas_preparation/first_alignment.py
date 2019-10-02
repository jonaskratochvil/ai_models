#!/usr/bin/env python3

from aeneas.executetask import ExecuteTask
from aeneas.task import Task
import numpy as np
from pydub.utils import which
import os
import json
import pandas as pd
from pydub import AudioSegment
import sys

"""
This main usage of this script is to perform the so called force alignment of a long audio file with the corresponding text.
This script expects for example 20 minutes long audio together with the text file in form where each line corresponds to one segment to which the algorithm
will split the audio and match the corresponding text.
The algorithm used comes from the Aeneas package and is based on signal processing techniques making it language dependent. It works for Czech with relatively high accuracy, however, some mistakes are to be expected especially in cases where the text does not match the audio content perfectly.
"""

arg = sys.argv[1].split(".")[0]

PATH = f"/home/jonas/Jonas-zaloha/Translate_vypisky/Ceskyrozhlas/Dvojka/final/{arg}"

config_string = "task_language=ces|is_text_type=plain|os_task_file_format=json"
task = Task(config_string=config_string)

# Path to the audio file
task.audio_file_path_absolute = f"{PATH}/{NAME}.wav"

# Path to the text file
task.text_file_path_absolute = f"{PATH}/final_ali.txt"

task.sync_map_file_path_absolute = f"{PATH}/output/syncmap.json"

ExecuteTask(task).execute()
task.output_sync_map_file()

# Alignment of text and audio

AudioSegment.converter = which("ffmpeg")

book = AudioSegment.from_file(f"{PATH}/{NAME}.wav")
with open(f"{PATH}/output/syncmap.json") as f:
    syncmap = json.loads(f.read())

sentences = []
for i in range(len(syncmap["fragments"])):
    sentences.append(
        {
            "audio": book[
                float(syncmap["fragments"][i]["begin"])
                * 1000: float(syncmap["fragments"][i]["end"])
                * 1000
            ],
            "text": syncmap["fragments"][i]["lines"][0],
        }
    )

# This puts together the audio segment and corresponding text
for i, sentence in enumerate(sentences):
    text = sentences[i]["text"]
    if len(text) == 0:
        continue

    sentences[i]["audio"].export(
        f"{PATH}/audio_segments/{NAME}_{str(i)}.wav", format="wav"
    )
    with open(f"{PATH}/audio_segments/{NAME}_{str(i)}.wav.trn", "w") as file1:
        print(f"{text}", file=file1)

