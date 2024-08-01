# download.py
import os
import pandas as pd
import shutil
import glob
import random

def download_mimii(data_root_dir: str, snr: str) -> None:
    """
    MIMII 데이터셋을 다운받는 함수

    Parameters
    ---------- 
    root: str
        데이터를 다운받고자 하는 루트 디렉토리 경로
    snr: str
        다운받고자 하는 snr 종류

    Returns
    ----------
        
    Examples
    ----------
    >>> df = download_mimii("../data", "0_dB")
    """

    data_dir = os.path.join(data_root_dir, snr)# 데이터셋을 저장할 디렉토리
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    phase_dir = os.listdir(data_dir)
    if "train" or "test" or "val" in phase_dir:
        return

    baseurl = "https://zenodo.org/records/3384388/files/"
    machinetypes = ["fan", "pump", "slider", "valve"]
    modelIDs = ["id_00", "id_02", "id_04", "id_06"]
    conditions = ["abnormal", "normal"]

    for machinetype in machinetypes:
        machinetype_dir = os.path.join(data_dir, machinetype)

        filename = os.path.join(data_dir, f"{machinetype}.zip") # 파일을 저장할 이름
        url = f"{baseurl}{snr}_{machinetype}.zip?download=1" # 그 때의 url

        # .wav파일 존재유무 확인
        if is_contain_wav_files(machinetype_dir):
            print(f".wav file already exist in the {machinetype} directory")
            return

        # zip 파일이 존재하지 않을 경우 다운로드
        if not os.path.isfile(filename): 
            os.system(f"wget -O {filename} {url}")
        else:
            print(f"{filename} already exist.")

        # 파일이 이미 압축 해제된 경우 확인
        if os.path.exists(machinetype_dir):
                print(f"{filename} already extracted.")
        else:
            os.system(f"unzip {filename} -d {data_dir}") # unzip 
        
        # 파일 이름 및 파일 위치 변경 
        for modelID in modelIDs:
            for condition in conditions:
                target_dir = os.path.join(machinetype_dir, modelID, condition)
                wav_files = os.listdir(target_dir)
                for wav_file in wav_files:
                    old_path = os.path.join(target_dir, wav_file)
                    new_name = f"{condition}_{modelID}_{wav_file}"
                    new_path = os.path.join(machinetype_dir, new_name)
                    shutil.move(old_path, new_path)

                    # unused 디렉토리 삭제
                    if not os.listdir(target_dir):
                        os.rmdir(target_dir)
                # unused 디렉토리 삭제
                condition_dir = os.path.join(machinetype_dir, modelID, condition)
                if os.path.exists(condition_dir) and not os.listdir(condition_dir):
                    os.rmdir(condition_dir)
            # unused 디렉토리 삭제
            modelID_dir = os.path.join(machinetype_dir, modelID)
            if os.path.exists(modelID_dir) and not os.listdir(modelID_dir):
                os.rmdir(modelID_dir)


def is_contain_wav_files(directory : str) -> bool:
    """
    .wav 파일이 있는지 검사하는 함수

    Parameters
    ---------- 
    directory: str
        .wav 파일이 있는지 검사하고자 하는 디렉토리

    Returns
    ----------
    bool
        .wav 파일이 있는지 하나라도 있으면 True, 그렇지 않다면 False
    """
    
    if not os.path.isdir(directory):
        print(f"{directory} doesn't exist.")
        return False
    
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            return True
    return False

def split_dirs(data_root_dir, snr):
    """
    다운받은 MIMII 데이터셋을 train, test, val 디렉토리로 분할하는 함수

    ex) split_dirs("../dataset", "../data", "0_dB")
    """

    data_dir = os.path.join(data_root_dir, snr)
    if not os.path.isdir(data_dir):
        print(f"{data_dir} doesn't exist. Download data first.")
        return

    machinetypes = ["fan", "pump", "slider", "valve"]
    modelIDs = ["id_00", "id_02", "id_04", "id_06"]
    for machinetype in machinetypes:
        machinetype_dir = os.path.join(data_dir, machinetype)
        # print(machinetype_dir)

        for phase in ["train", "test", "val"]:
            phase_dir = os.path.join(machinetype_dir, phase)
            if not os.path.isdir(phase_dir):
                os.makedirs(phase_dir)

        files = os.listdir(machinetype_dir)

        for file in files:
            # make test dir
            if "id_06" in file:
                src_file = os.path.join(machinetype_dir, file)
                dst_file = os.path.join(machinetype_dir, "test", file)
                shutil.move(src_file, dst_file)
        
        for modelID in modelIDs:
            move_to_val_dir(machinetype_dir, modelID)   

        files = [f for f in os.listdir(machinetype_dir) if f not in ["val", "train", "test"]]

        for file in files:
            src_file = os.path.join(machinetype_dir, file)
            dst_file = os.path.join(machinetype_dir, "train", file)
            shutil.move(src_file, dst_file)


def move_to_val_dir(directory, modelID):
    random.seed(42)
    all_wav_files = glob.glob(os.path.join(directory, '*.wav'))
    abnormal_files = sorted([f for f in all_wav_files if 'abnormal' in f and modelID in f])
    # print(abnormal_files)
    num_abnormal_files = len(abnormal_files)

    normal_files = sorted([f for f in all_wav_files if 'normal' in f and modelID in f and 'abnormal' not in f])
    selected_normal_files = random.sample(normal_files, min(num_abnormal_files, len(normal_files)))

    for file in abnormal_files:
        src_file =  file
        # print(src_file)
        dst_file = os.path.join(directory, "val")
        # print(dst_file)
        shutil.move(src_file, dst_file)

    for file in selected_normal_files:
        src_file =  file
        # print(src_file)
        dst_file = os.path.join(directory, "val")
        # print(dst_file)
        shutil.move(src_file, dst_file)