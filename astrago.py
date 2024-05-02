from tqdm import tqdm
from torchsummary import summary
import xml.etree.ElementTree as ET
import subprocess
import psutil
import torch
import glob
import json
import yaml
import time
import csv
import os


class Astrago(tqdm):
    model_name = ''
    param = 0
    gpu = ''
    FLOPS = 14
    data_info = 0
    image_size = 0
    batch_size = 0
    train_time = 0
    val_time = 0
    csv_file_path = ""

    
    # csv 파일 저장 함수
    def save_metrics_to_csv(model_name, param, gpu, FLOPS, t_img_num, v_img_num, imgsz, batch, epoch,
                            train_time, val_time, epoch_time, elapsed, remaining):
        '''
            model_name : model 이름
            param : model 파라미터 수
            gpu : gpu 종류
            FLOPS : gpu flops
            t_img num : train 이미지 수
            v_img num : validation 이미지 수
            imgsz : input 이미지 사이즈 (pixel)
            batch : 배치 사이즈
            epoch : 에폭 수
            train_time : 학습하는 데 걸린 시간 (sec)
            val_time : 검증하는 데 걸린 시간 (sec)
            epoch_time : 에폭 처리 시간 (sec)
            elapsed : 누적 학습 시간 (sec)
            remaining : 예상 남은 시간 (sec)
        '''
        Astrago.csv_file_path = f'/workspace/{Astrago.model_name}_i{Astrago.image_size}_b{Astrago.batch_size}.csv'
        with open(Astrago.csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            if file.tell() == 0:
                writer.writerow(['model_name', 'param', 'gpu', 'FLOPS', 't_img_num', 'v_img_num', 'imgsz', 'batch', 
                                 'epoch', 'train_time', 'val_time', 'epoch_time', 'elapsed', 'remaining'])

            writer.writerow([model_name, param, gpu, FLOPS, t_img_num, v_img_num, imgsz, batch, epoch,
                             train_time, val_time, epoch_time, elapsed, remaining])
    
    
    
    @staticmethod
    def format_meter(n, total, elapsed, rate=None, initial=0, *args, **kwargs):
        torch.cuda.synchronize()
        t = Astrago.format_interval((total - n) / rate) if rate else 0
        t = Astrago.time_to_seconds(str(t))
        epoch_per_time = (elapsed / n) if n > 0 else 0
        d_info = Astrago.data_info
        t_img_num = d_info[0] 
        v_img_num = d_info[1]
        imgsz = Astrago.image_size
        batch = Astrago.batch_size
        train_time = Astrago.train_time
        val_time = Astrago.val_time

        
        # 마지막 에폭만 두 번 출력되고 저장되는 경우가 있어 임시 해결책
        if n == total:
            if not hasattr(Astrago, 'last_epoch_done') or not Astrago.last_epoch_done:  # 마지막 에폭이 한 번만 출력되도록 확인하는 변수
                Astrago.save_metrics_to_csv(Astrago.model_name, Astrago.param, Astrago.gpu, Astrago.FLOPS, t_img_num, 
                                            v_img_num, imgsz, batch, n, train_time, val_time, epoch_per_time, elapsed, t)
                
                print(f'\n현재 epoch > {n}/{total}')
                print(f'train 시간 > {train_time}')
                print(f'validation 시간 > {val_time}')
                print(f'처리 시간 > {epoch_per_time}s')
                print(f'종료까지 남은 예상 시간 > {t}')
                print(f'현재까지 작업에 소요된 시간 > {elapsed}s')
                Astrago.last_epoch_done = True  # 마지막 에폭이 출력되었음을 표시
                

        else:  # 마지막 에폭이 아닐 때는 출력 및 저장을 수행
            Astrago.save_metrics_to_csv(Astrago.model_name, Astrago.param, Astrago.gpu, Astrago.FLOPS, t_img_num, 
                                        v_img_num, imgsz, batch, n, train_time, val_time, epoch_per_time, elapsed, t)
            
            print(f'\n현재 epoch > {n}/{total}')
            print(f'train 시간 > {train_time}')
            print(f'validation 시간 > {val_time}')
            print(f'처리 시간 > {epoch_per_time}s')
            print(f'종료까지 남은 예상 시간 > {t}')
            print(f'현재까지 작업에 소요된 시간 > {elapsed}s')
                

        return tqdm.format_meter(n, total, elapsed, rate=rate, initial=initial, *args, **kwargs)
    
    
    def get_elapsed_train_time(t_time):
        Astrago.train_time = t_time
        
    def get_elapsed_val_time(v_time):
        Astrago.val_time = v_time

    

    # 모델 train 시 사용하고 있는 gpu 정보
    def get_gpu_info():
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        Astrago.gpu = gpu_name
    
    
    # 시간 변환 함수
    def time_to_seconds(time_str):
        components = [int(x) for x in time_str.split(':')]
        while len(components) < 3:
            components.insert(0, 0)

        hours, minutes, seconds = components
        
        return hours * 3600 + minutes * 60 + seconds
        
    
    # 모델 이름 추출 함수
    def get_model_name(model):
        Astrago.model_name = model
        return model
        
        
    # 모델 파라미터 수 추출 함수    
    def get_model_params(model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f'\n[Params]')
        print(f'파라미터 수 : {total_params}')
        Astrago.param = total_params
        
    
    
    # 모델 train 시, input image 사이즈 추출 함수     
    def get_image_size(image_size_argument):
        image_size = image_size_argument
        print(f'\n[Image Size]')
        print(f'이미지 사이즈 : {image_size}')
        Astrago.image_size = image_size
        return image_size
    
    
    # 모델 train 시, 배치 사이즈 추출 함수
    def get_batch_size(batch_argument):
        batch_size = batch_argument
        print(f'\n[Batch Size]')
        print(f'배치 사이즈 : {batch_size}')
        Astrago.batch_size = batch_size
        
        
    def count_data_num(data_path):
        cnt = 0
        
        for folder in os.listdir(data_path):
            cnt+=len(os.listdir(os.path.join(data_path, folder)))
        
        return cnt
    
        
    # UNet 데이터 수량 check    
    def get_data_info(data_path):
        train_dir = os.path.join(data_path, 'train')
        val_dir = os.path.join(data_path, 'val')
        train_num = Astrago.count_data_num(train_dir)
        val_num = Astrago.count_data_num(val_dir)
        print(f'\n[Train DATA]')
        print(f'DATA NUM : {train_num}')
        print(f'\n[Val DATA]')
        print(f'DATA NUM : {val_num}')
        Astrago.data_info = [train_num, val_num]
        
    
  
    
            
            
            
            
            
class Inference(tqdm):
    
    model_name = ''
    param = 0
    gpu = ''
    FLOPS = 14
    data_num = 0
    image_size = 0
    inference_time = 0
    save_time = 0
    single_data_inference_time = 0
    csv_file_path = ''

    
    # csv 파일 저장 함수
    def save_metrics_to_csv(model_name, param, gpu, FLOPS, data_num, imgsz, 
                            num, inference_time, save_time, single_data_inference_time):
        '''
            model_name : model 이름
            param : model 파라미터 수
            gpu : gpu 종류
            FLOPS : gpu flops
            data_num : 예측할 이미지 수
            imgsz : input 이미지 사이즈 (pixel)
            num : 처리 중인 이미지(프레임) 순번
            inference_time : inference 하는 데만 걸린 시간
            save_time : inference 결과 저장하는 데 걸린 시간
            single_data_inference_time : 이미지 하나 당 inference 속도 (sec) == inference_time + save_time
        '''
        Inference.csv_file_path = f'/workspace/{Inference.model_name}_i{Inference.image_size}_inference.csv'
        with open(Inference.csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            if file.tell() == 0:
                writer.writerow(['model_name', 'param', 'gpu', 'FLOPS', 'data_num','imgsz', 
                                 'num', 'inference_time', 'save_time', 'single_data_inference_time'])

            writer.writerow([model_name, param, gpu, FLOPS, data_num, imgsz, 
                             num, inference_time, save_time, single_data_inference_time])
    
    
    
    @staticmethod
    def format_meter(n, total, elapsed, rate=None, initial=0, *args, **kwargs):
        torch.cuda.synchronize()
        data_num = Inference.data_num
        imgsz = Inference.image_size
        inference_time = Inference.inference_time
        save_time = Inference.save_time
        single_data_inference_time = inference_time + save_time

        if total != 1:
        
            # 마지막 에폭만 두 번 출력되고 저장되는 경우가 있어 임시 해결책
            if n == total:
                if not hasattr(Inference, 'last_epoch_done') or not Inference.last_epoch_done:  # 마지막 에폭이 한 번만 출력되도록 확인하는 변수
                    Inference.save_metrics_to_csv(Inference.model_name, Inference.param, Inference.gpu, Inference.FLOPS, 
                                                  data_num, imgsz, n, inference_time, save_time, single_data_inference_time)
                    
                    print(f'\n현재 진행 순번> {n}/{total}')
                    print(f'inference time > {inference_time}')
                    print(f'save time > {save_time}')
                    print(f'이미지 1장 당 추론 시간 > {single_data_inference_time}')
                    Inference.last_epoch_done = True  # 마지막 에폭이 출력되었음을 표시
                    

            else:  # 마지막 에폭이 아닐 때는 출력 및 저장을 수행
                Inference.save_metrics_to_csv(Inference.model_name, Inference.param, Inference.gpu, Inference.FLOPS, 
                                              data_num, imgsz, n, inference_time, save_time,  single_data_inference_time)
                    
                print(f'\n현재 진행 순번> {n}/{total}')
                print(f'inference time > {inference_time}')
                print(f'save time > {save_time}')
                print(f'이미지 1장 당 추론 시간 > {single_data_inference_time}')
                    

        return tqdm.format_meter(n, total, elapsed, rate=rate, initial=initial, *args, **kwargs)
    
    
    # 모델 predict 시 inference 시간 추출 함수
    def get_elapsed_inference_time(start_time):
        Inference.inference_time = time.time() - start_time
        
    
    # 모델 predict 시 inference 결과 저장 시간 추출 함수
    def get_elapsed_save_time(start_time):
        Inference.save_time = time.time() - start_time
        
        
    # predict 이미지/프레임 수 추출 함수
    def get_data_num(data_info_argument):
        data_num = len(data_info_argument)
        print(f'\n[Data]')
        print(f'데이터 수 : {data_num}')
        Inference.data_num = data_num
    
    
    # 시간 변환 함수
    def time_to_seconds(time_str):
        components = [int(x) for x in time_str.split(':')]
        while len(components) < 3:
            components.insert(0, 0)

        hours, minutes, seconds = components
        
        return hours * 3600 + minutes * 60 + seconds
        
        
    # 모델 파라미터 수 추출 함수    
    def get_model_params(model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f'\n[Params]')
        print(f'파라미터 수 : {total_params}')
        Inference.param = total_params
        
    
    
    # 모델 predict 시, input image 사이즈 추출 함수     
    def get_image_size(image_size_argument):
        image_size = image_size_argument
        print(f'\n[Image Size]')
        print(f'이미지 사이즈 : {image_size}')
        Inference.image_size = image_size
    
    
    
        
        
    