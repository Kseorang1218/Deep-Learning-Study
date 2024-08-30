# download.py
import os

import zipfile
from tqdm import tqdm

import pandas as pd
import librosa

MACHINETYPE_LIST = ["fan", "pump", "slider", "valve"]
MODELID_LIST = ["id_00", "id_02", "id_04", "id_06"]
FAULTTYPE_LIST = ["abnormal", "normal"]
PHASE_LIST = ["train", "test", "val"]

def download_mimii(data_root_dir: str, snr: str, sample_rate: int) -> pd.DataFrame:
    """
    MIMII 데이터셋을 다운받는 함수

    Parameters
    ---------- 
    data_root_dir: str
        데이터를 다운받고자 하는 루트 디렉토리 경로
    snr: str
        다운받고자 하는 snr 종류
    sample_rate: int
        오디오 파일을 불러올 때 사용할 샘플링 레이트

    Returns
    ----------
    pd.DataFrame
        data, model_id, fault_type, machine_type을 열으로 하는 데이터프레임.
        
    Examples
    ----------
    >>> df = download_mimii("./data", "0_dB", 16000)
    """

    df = {
        'data': [],
        'model_id': [],
        'fault_type': [],
        'machine_type': []
    }

    data_dir = os.path.join(data_root_dir, snr)# 데이터셋을 저장할 디렉토리
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    baseurl = "https://zenodo.org/records/3384388/files/"

    for machinetype in MACHINETYPE_LIST:
        machinetype_dir = os.path.join(data_dir, machinetype)

        filename = os.path.join(data_dir, f"{machinetype}.zip") # 파일을 저장할 이름
        url = f"{baseurl}{snr}_{machinetype}.zip?download=1"

        # zip 파일이 존재하지 않을 경우 다운로드
        if not os.path.isfile(filename): 
            os.system(f"wget -O {filename} {url}")
        else:
            print(f"{filename} already exist.")

        if not os.path.exists(machinetype_dir):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), desc=f"Extracting {machinetype}"):
                    zip_ref.extract(file, data_dir)
        else:
            print(f"{filename} already extracted.")

            
    for machine_type in MACHINETYPE_LIST:
        for model_id in MODELID_LIST:
            for fault_type in FAULTTYPE_LIST:
                target_dir = os.path.join(data_root_dir, snr, machine_type, model_id, fault_type)
                with os.scandir(target_dir) as files:
                    for file in tqdm(files, desc=f"Processing {machine_type} - {model_id} - {fault_type}"):
                        data, _ = librosa.load(file, sr=sample_rate)
                        df['data'].append(data)
                        df['model_id'].append(model_id)
                        df['fault_type'].append(fault_type)
                        df['machine_type'].append(machine_type)

    df = pd.DataFrame(df)
    
    return df
