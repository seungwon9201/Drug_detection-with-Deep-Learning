import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import torch
import tkinter as tk
from tkinter import messagebox
import numpy as np
from tkinter import ttk
from tkinter import Canvas
import threading
import time
import csv
import os

# 미리 학습된 모델을 불러옵니다.
model = joblib.load('C:/Users/ey896/OneDrive/Desktop/xgboost_model_pca_RandomHyper.pkl')
# BERTweet 토크나이저와 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
bertweet_model = AutoModel.from_pretrained('vinai/bertweet-base')
# 미리 학습된 PCA 모델 불러오기
pca = joblib.load('C:/Users/ey896/OneDrive/Desktop/pca_model (1).pkl')

# 입력된 텍스트를 모델에 맞게 전처리합니다.
# 전처리 함수는 모델을 학습할 때 사용한 방식에 맞추어야 합니다.
def preprocess(text):
    # 예시 전처리 (필요에 따라 수정)
    text = text.lower()
    return text

# 탐지 결과 저장 함수
def save_result(user_input, prediction):
    file_exists = os.path.isfile('detection_results.csv')
    with open('detection_results.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Input Text', 'Prediction'])  # 헤더 작성
        writer.writerow([user_input, prediction])

# UI 구현
root = tk.Tk()
root.title("마약 판매 게시글 탐지기")
root.geometry("500x300")  # 세로 길이를 줄임
root.configure(bg='#2b2b2b')

style = ttk.Style()
style.theme_use('clam')
style.configure("TButton", foreground='#ffffff', background='#4a90e2', font=('Helvetica', 12, 'bold'))
style.configure("TLabel", foreground='#ffffff', background='#2b2b2b', font=('Helvetica', 12))
style.configure("TEntry", fieldbackground='#3c3f41', foreground='#ffffff')

# 로딩 애니메이션 함수
def start_loading(canvas, arc):
    for i in range(0, 360, 10):
        canvas.itemconfig(arc, start=i)
        root.update_idletasks()
        time.sleep(0.05)
    canvas.destroy()

# 탐지 함수
def detect():
    result_label.config(text="")
    user_input = entry.get()
    if not user_input.strip():
        messagebox.showwarning("입력 오류", "텍스트를 입력하세요.")
        return
    
    disable_button()
    
    # 로딩 애니메이션을 위한 캔버스 생성
    canvas = Canvas(root, width=50, height=50, bg='#2b2b2b', highlightthickness=0)
    canvas.pack(pady=10)
    arc = canvas.create_arc(10, 10, 40, 40, start=0, extent=150, outline='#4a90e2', width=5)
    
    # 로딩 애니메이션을 스레드로 실행
    loading_thread = threading.Thread(target=start_loading, args=(canvas, arc))
    loading_thread.start()
    
    # 탐지 작업을 스레드로 실행
    detection_thread = threading.Thread(target=run_detection, args=(user_input, loading_thread))
    detection_thread.start()

# 탐지 작업 함수
def run_detection(user_input, loading_thread):
    loading_thread.join()  # 로딩이 완료될 때까지 대기
    
    preprocessed_text = preprocess(user_input)
    try:
        # BERTweet을 이용해 텍스트 임베딩 생성
        tokens = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        if tokens['input_ids'].size(1) > bertweet_model.config.max_position_embeddings:
            messagebox.showwarning("입력 오류", "입력 텍스트가 너무 깁니다. 더 짧게 입력해 주세요.")
            enable_button()
            return
        
        with torch.no_grad():
            embeddings = bertweet_model(**tokens).last_hidden_state.mean(dim=1)
            # PCA를 사용하여 차원 축소
            embeddings_reduced = pca.transform(embeddings.numpy())

        # 예측 수행
        prediction = model.predict(embeddings_reduced)

        # 예측 결과 처리
        if prediction == 0:
            result = "마약 관련 공익 게시글입니다."
            result_label.config(text=result, foreground='green')
        elif prediction == 1:
            result = "마약 판매 게시글입니다."
            result_label.config(text=result, foreground='red')
        elif prediction == 2:
            result = "일반 게시글입니다."
            result_label.config(text=result, foreground='green')
        
        # 결과 저장
        save_result(user_input, result)
    except Exception as e:
        messagebox.showerror("오류 발생", f"오류가 발생했습니다: {str(e)}")
    finally:
        enable_button()
    
    # 입력 필드 초기화
    entry.delete(0, 'end')

# 버튼 비활성화 및 활성화 함수
def disable_button():
    button.config(state=tk.DISABLED)

def enable_button():
    button.config(state=tk.NORMAL)

# 라벨과 입력 필드 생성
label = ttk.Label(root, text="게시글 텍스트를 입력하세요:")
label.pack(pady=10)

entry = ttk.Entry(root, width=60)
entry.pack(pady=5)

# 탐지 버튼 생성
button = ttk.Button(root, text="탐지하기", command=detect)
button.pack(pady=20)

# 결과 라벨 생성
result_label = ttk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()