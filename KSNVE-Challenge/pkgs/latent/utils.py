# utils.py

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="KSNVE challenge")
    
    # latent space 인자
    parser.add_argument(
        "--latent_size_list", 
        type=int, 
        nargs="+", 
        default=[4096, 2048, 1024, 512],
        help="latent space 입력 (예: --latent_size_list 4096 2048 1024 512, 기본값: [4096, 2048, 1024, 512])"
    )
        
    parser.add_argument(
        "--latent_size",
        type=int,
        default=512,
        help="단일 latent space 크기 (예: --latent_size 512, 기본값: 512)"
    )
    
    # 새로운 인자 추가
    parser.add_argument(
        "--z",
        action="store_true",
        help="z 옵션 활성화 (예: --z, 기본값: 비활성화(False))"
    )
    
    parser.add_argument(
        "--DSVDD",
        action="store_true",
        help="DSVDD 옵션 활성화 (예: --DSVDD, 기본값: 비활성화(False))"
    )
    
    parser.add_argument(
        "--FFT",
        action="store_true",
        help="FFT 옵션 활성화 (예: --FFT, 기본값: 비활성화(False))"
    )
    
    args = parser.parse_args()

    return args
