import os
import pandas as pd
import librosa

def download_mimii(root: str, snr: str) -> pd.DataFrame:
    """
    MIMII 데이터셋을 다운받고 경로, 기계종류, 모델id, 레이블을 rows로 하는 데이터프레임을 반환하는 함수

    Parameters
    ---------- 
    root: str
        데이터를 다운받고자 하는 루트 디렉토리 경로
    snr: str
        다운받고자 하는 snr 종류

    Returns
    ----------
    pd.DataFrame
        경로, 기계종류, 모델id, 레이블을 rows로 하는 데이터프레임

        
    Examples
    ----------
    >>> df = download.download_mimii("../../data/mimii", "0_dB")
    """

    filedir = root + "/" + snr # 파일 저장할 디렉토리
    baseurl = "https://zenodo.org/records/3384388/files/"
    machinetypes = ["fan", "pump", "slider", "valve"]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    for machinetype in machinetypes:
        filename = filedir + "/" + machinetype + ".zip" # 파일을 저장할 이름
        url = baseurl + snr + "_" + machinetype + ".zip?download=1" # 그 때의 url

        if not os.path.isfile(filename): # zip 파일 다운로드
            os.system(f"wget -O {filename} {url}")
        else:
            print(f"{filename} already exist.")

        if os.path.exists(filedir + "/" + machinetype + "/"):
                print(f"{filename} already extracted.")
        else:
            os.system(f"unzip {filename} -d {filedir}") # unzip 



    label_map = {
        ("fan", "normal"): 0,
        ("pump", "normal"): 0,
        ("slider", "normal"): 0,
        ("valve", "normal"): 0,
        ("fan", "abnormal"): 1,
        ("pump", "abnormal"): 2,
        ("slider", "abnormal"): 3,
        ("valve", "abnormal"): 4,
    }

    modelIDs = ["id_00", "id_02", "id_04", "id_06"]
    faulttypes = ["abnormal", "normal"]

    df = {
        "data": [],
        "samplerate": [],
        "path": [],
        "machinetype": [],
        "modelID": [],
        "label": []
    }

    for machinetype in machinetypes:
        for modelID in modelIDs:
            for faulttype in faulttypes:
                with os.scandir(f"{filedir}/{machinetype}/{modelID}/{faulttype}") as files:
                    for file in files:
                        data, sample_rate= librosa.load(file, sr=16000)
                        df["data"].append(data)
                        df["samplerate"].append(sample_rate)
                        df["path"].append(file.path)
                        df["machinetype"].append(machinetype)
                        df["modelID"].append(modelID)
                        df["label"].append(label_map[(machinetype, faulttype)])

    dataframe = pd.DataFrame(df)

    return dataframe


