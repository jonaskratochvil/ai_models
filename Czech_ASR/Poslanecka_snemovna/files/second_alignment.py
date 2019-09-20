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

DIR = sys.argv[1]
NAME = sys.argv[2]


#arg = sys.argv[1].split(".")[0]
#NAME = arg

PATH = f"/home/jonas/Jonas-zaloha/Translate_vypisky/Ceskyrozhlas/Poslanecka_Snemovna/files/{DIR}/{NAME}"

#spk_id = []
# with open(f"{PATH}/speaker_id.txt", "r") as data:
#    for line in data:
#        spk_id.append(line.rstrip())

config_string = "task_language=ces|is_text_type=plain|os_task_file_format=json"
task = Task(config_string=config_string)

task.audio_file_path_absolute = f"{PATH}/{NAME}.wav"

task.text_file_path_absolute = f"{PATH}/{NAME}.txt"


task.sync_map_file_path_absolute = f"{PATH}/output/syncmap.json"

ExecuteTask(task).execute()
task.output_sync_map_file()

# First alignment

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

for i, sentence in enumerate(sentences):
    text = sentences[i]["text"]
    #spk = spk_id[i]
    if len(text) == 0:
        continue

    sentences[i]["audio"].export(
        f"{PATH}/audio_segments/{NAME}_{str(i)}.wav", format="wav"
    )
    with open(f"{PATH}/audio_segments/{NAME}_{str(i)}.wav.trn", "w") as file1:
        print(f"{text}", file=file1)

    # with open(f"{PATH}/audio_segments/{NAME}_{str(i)}.wav.spk.trn", "w") as file2:
    #    print(f"{spk}", file=file2)
