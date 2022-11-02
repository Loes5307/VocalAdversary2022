########################################################################################
#
# This script can be used to generate the CSV file(s) which will be used to write
# the VoxCeleb dataset into shards.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib
import re

from typing import Tuple, Optional, Set

import click
import pandas as pd
import torchaudio
import yaspin

from tqdm import tqdm

from data_utility.eval.speaker.evaluator import SpeakerTrial
from data_utility.util.csv import ShardCsvSample

########################################################################################
# logic for traversing dataset folder and writing info to CSV file


def load_speaker_meta(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        names=["voxceleb_id", "vggface_id", "gender", "nationality", "set", "dataset"],
    )

    return df


def load_trials_as_sample_id_set(path: pathlib.Path):
    utterance_set = set()

    for trial in SpeakerTrial.from_file(path):
        utterance_set.add(trial.left)
        utterance_set.add(trial.right)

    return utterance_set


def traverse_split(
    path: Tuple[pathlib.Path],
    df_speaker_meta: pd.DataFrame,
    extension: str,
    utterance_filter: Optional[Set] = None,
):
    all_samples = []
    all_speaker_ids = set(df_speaker_meta["voxceleb_id"].unique().tolist())

    potential_dirs = []

    for p in path:
        potential_dirs.extend([d for d in p.iterdir()])

    # we search for each child directory of `path` that is a valid speaker_id
    for speaker_dir in tqdm(potential_dirs):
        if speaker_dir.name not in all_speaker_ids:
            print("skipping....")
            continue

        speaker_id = speaker_dir.name
        speaker_info = df_speaker_meta.loc[df_speaker_meta["voxceleb_id"] == speaker_id]
        gender = speaker_info["gender"].item()

        dataset_id_string = speaker_info["dataset"].item()
        if dataset_id_string == "vox2":
            dataset_id = "vc2"
        elif dataset_id_string == "vox1":
            dataset_id = "vc1"
        else:
            raise ValueError(f"unknown {dataset_id_string=}")

        # we search for each child directory that contains wav files
        for recording_dir in speaker_dir.iterdir():
            all_audio_files = [
                f
                for f in recording_dir.iterdir()
                if f.is_file() and extension in f.suffix
            ]

            if len(all_audio_files) == 0:
                continue

            recording_id = recording_dir.name

            # create sample for each audio file
            for audio_file in all_audio_files:
                utterance_id = audio_file.stem

                key = f"{dataset_id}/{speaker_id}/{recording_id}/{utterance_id}"

                if utterance_filter is not None and key not in utterance_filter:
                    continue

                meta = torchaudio.info(audio_file)

                sample = ShardCsvSample(
                    key=key,
                    path=str(audio_file),
                    num_frames=meta.num_frames,
                    sample_rate=meta.sample_rate,
                    transcription=None,
                    speaker_id=f"{dataset_id}/{speaker_id}",
                    recording_id=f"{dataset_id}/{recording_id}",
                    gender=gender,
                )

                all_samples.append(sample)

    if len(all_samples) == 0:
        raise ValueError(f"unable to find any samples in {[str(p) for p in path]}")

    return ShardCsvSample.to_dataframe(all_samples)


########################################################################################
# entrypoint of script


@click.command()
@click.argument(
    "dir_path",
    type=pathlib.Path,
    nargs=-1,
    required=True,
)
@click.option(
    "--csv",
    "csv_file",
    type=pathlib.Path,
    required=True,
    help="path to write output csv file to",
)
@click.option(
    "--meta",
    "meta_csv_file",
    type=pathlib.Path,
    required=True,
    help="path to CSV file containing metadata on speaker IDs",
)
@click.option(
    "--trials",
    "trial_csv_path",
    type=pathlib.Path,
    required=False,
    help="path to text file containing trials of samples. "
    "When given, only keys in this file will be written to csv file",
)
@click.option(
    "--ext",
    "extension",
    type=str,
    default="wav",
    help="the extension used for each audio file",
)
def main(
    dir_path: Tuple[pathlib.Path],
    csv_file: pathlib.Path,
    meta_csv_file: pathlib.Path,
    extension: str,
    trial_csv_path: Optional[pathlib.Path] = None,
):
    print(
        f"Generating {str(csv_file)} with data in {[str(p) for p in dir_path]}",
        flush=True,
    )
    meta = load_speaker_meta(meta_csv_file)

    if trial_csv_path is not None:
        utterance_filter = load_trials_as_sample_id_set(trial_csv_path)
    else:
        utterance_filter = None

    df_found = traverse_split(
        path=dir_path,
        df_speaker_meta=meta,
        extension=extension,
        utterance_filter=utterance_filter,
    )

    with yaspin.yaspin(text=f"writing {csv_file}"):
        df_found.to_csv(str(csv_file), index=False)


if __name__ == "__main__":
    main()
