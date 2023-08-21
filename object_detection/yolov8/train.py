import argparse
import os
import logging
from datetime import datetime
import boto3
import sys
import pandas as pd
import shutil
from sagemaker_training import environment
from ultralytics import YOLO

# --- Logging
logging.basicConfig(level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train(args):
    
    logger.info(f'Current path is {os.getcwd()}')
    
    # Store base model and training job name in seperate variable such that remaining dict can be used as training parameters
    training_job_name = args['training_job_name']
    base_model = args['base_model']
    del args['training_job_name']
    del args['base_model']
    
    # Check if model training should be started or resumed (for running training on spot instance)
    if not os.listdir('/opt/ml/checkpoints'):
        # Start new model training
        logger.info('Starting new training run')
        
        # Load a model
        model = YOLO(base_model)  # load a pretrained model (recommended for training)
        
        # Add callback that saves training files to checkpoint folder
        model.add_callback("on_train_epoch_start", copy_train_checkpoint_files)
        model.add_callback("on_train_end", copy_train_checkpoint_files)

        # Train the model
        model.train(**args)
    else:
        # Resume model training
        logger.info('Continuing training run')
        
        # Copy over checkpoint files to runs folder (such that new metrics will be added to old ones when spot instance is terminated)
        shutil.copytree('/opt/ml/checkpoints', '/opt/ml/code/runs/detect/train', dirs_exist_ok=True)
        
        # Load last model
        try:
            model = YOLO('/opt/ml/code/runs/detect/train/weights/last.pt')
        # Load last model for hyperparameter tuning job
        except:
            env = environment.Environment()
            job_name = env["job_name"]
            model = YOLO(f'/opt/ml/code/runs/detect/train/{job_name}/weights/last.pt')
        
        # Add callback that saves training files to checkpoint folder
        model.add_callback("on_train_epoch_start", copy_train_checkpoint_files)
        model.add_callback("on_train_end", copy_train_checkpoint_files)

        # Resume training
        model.train(resume=True)
    
    # Log training metrics
    df_results = pd.read_csv('/opt/ml/code/runs/detect/train/results.csv')
    idx_max_map50_95 = df_results['    metrics/mAP50-95(B)'].idxmax()
    row_best = df_results.loc[[idx_max_map50_95]]
    
    metric_epoch = int(row_best['                  epoch'])
    metric_precision = float(row_best['   metrics/precision(B)'])
    metric_recall = float(row_best['      metrics/recall(B)'])
    metric_map50 = float(row_best['       metrics/mAP50(B)'])
    metric_map50_95 = float(row_best['    metrics/mAP50-95(B)'])
    
    logger.info(f'Epoch_best={metric_epoch}; Precision={metric_precision}; Recall={metric_recall}; mAP@.5={metric_map50}; mAP@.5:.95={metric_map50_95};')


def copy_train_checkpoint_files(trainer):
    # Copy training files to checkpoints folder which will be save to S3 when training job is terminated
    shutil.copytree('/opt/ml/code/runs/detect/train', '/opt/ml/checkpoints', dirs_exist_ok=True)


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    # Desriptions of the arguments used for the ultralytics yolov8 training can be found here: https://docs.ultralytics.com/cfg/
    parser.add_argument('--base_model', type=str, default='yolov8l.pt')
    parser.add_argument('--training-job-name', type=str, default='checkpoints-yolov8-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    
    parser.add_argument('--data', type=str, default='data.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--save_period', type=int, default=-1)
    parser.add_argument('--image_weights', type=bool, default=False)
    parser.add_argument('--rect', type=bool, default=False)
    parser.add_argument('--close_mosaic', type=int, default=10)
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--freeze', type=int, default=0)

    
    parser.add_argument('--degrees', type=float, default=0.0)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--shear', type=float, default=0.0)
    parser.add_argument('--perspective', type=float, default=0.0)
    parser.add_argument('--flipud', type=float, default=0.0)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--copy_paste', type=float, default=0.1)
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    logger.info(f'Arguments after parsing is: \n {args_dict}')
    logger.info(f'Start training')
    
    train(args=args_dict)
    
    logger.info(f'Completed training')