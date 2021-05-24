# %%
from comet_ml import Experiment

import os
import time
import copy
import datetime
import pytz

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import *

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import split_dataset_into_3
from evaluator import *

# %%
# 学習ループ
def train_simple_model(model, dataloaders, class_names, device, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

# %%
# test画像で検証 -> 可視化
def visualize_model(model, dataloaders, class_names, device, imshow, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        correct = 0
        total = 0
        
        for idx, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            correct += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
        print('Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
        
        for j in range(inputs.size()[0]):
            images_so_far += 1
            plt.subplots_adjust(wspace=0.4, hspace=1.0)
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('Predicted: {}\nGround Truth: {}'.format(class_names[preds[j]], class_names[labels[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
                
        model.train(mode=was_training)

# 学習ループ：metricsとtensorboard出力あり
def train_model_metrics(model, dataloaders, class_names, device, criterion, optimizer, scheduler, num_epochs=25, save_model_name='binary_model', save_tensorboard_name='binary_runs'):
    writer = SummaryWriter('tensorboard_runs/{}/{}'.format(save_tensorboard_name, save_model_name))
    save_model_dir = 'save_models/{}'.format(save_model_name)
    os.makedirs(save_model_dir, exist_ok=True)
    d = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    save_day = '{}_{}{}_{}-{}'.format(d.year, d.month, d.day, d.hour, d.minute)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_precision = 0.0
    
    task = 'binary'
    
    if len(class_names) == 2:
        print('Task: Binary Class')
        task = 'binary'
    else:
        print('Task: Multi Class')
        task = 'multi'

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
        
            labels_all = []
            pred_all = []
            
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    axis = 1
                    _, preds = torch.max(outputs, axis)
                    loss = criterion(outputs, labels) 
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                                        
                # 学習の評価＆統計
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                pred_all.extend(predict.item() for predict in preds)
                labels_all.extend(label.item() for label in labels)
                            
            if phase == 'train':
                scheduler.step()
                if epoch%10 == 0:
                    torch.save(model.state_dict(), os.path.join(save_model_dir, save_model_name+'_{}_{}.pkl'.format(epoch, save_day)))
                    print('saving model epoch :{}'.format(epoch))
                    
            # 評価項目 (loss, accracy, recall, precision, f1-score)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            metrics_dict = classification_report(y_true=labels_all, y_pred=pred_all, output_dict=True, zero_division=0)

            epoch_recall = metrics_dict['macro avg']['recall']
            epoch_precision = metrics_dict['macro avg']['precision']
            epoch_f1 = metrics_dict['macro avg']['f1-score']

            plot_image_array = plot_cm(labels_all, pred_all, class_names)
            plot_image_roc = plot_roc(labels_all, pred_all, class_names, task)
            
            writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
            writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, epoch)
            writer.add_scalar('Recall/{}'.format(phase), epoch_recall, epoch)
            writer.add_scalar('Precision/{}'.format(phase), epoch_precision, epoch)
            writer.add_scalar('F1-score/{}'.format(phase), epoch_f1, epoch)
            writer.add_image('Confusion Matrix/{}'.format(phase), plot_image_array, epoch)
            writer.add_image('ROC Curve/{}'.format(phase), plot_image_roc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f} Recall: {:.4f} Precision: {:.4f} F1-score: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_recall, epoch_precision, epoch_f1))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                if epoch_recall==1 and epoch_precision > best_precision:
                    torch.save(model.state_dict(), 
                               os.path.join(save_model_dir, save_model_name+'_{}_{}_recall_1.0.pkl'.format(epoch, save_day)))
                    print('saving model recall=1.0 epoch :{}'.format(epoch))
                    recall_1_precision = epoch_precision
                best_precision = epoch_precision
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}, Precision: {:.4f}'.format(best_acc, best_precision))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 
               os.path.join(save_model_dir, save_model_name+'_{}_{}_best.pkl'.format(epoch, save_day)))
    writer.close()
    
    return model


# 学習ループ（comet.mlへ転送）
def train_model_cometml(experiment, hyper_params, confusion_matrix, model, dataloaders, class_names, device, criterion, optimizer, scheduler, save_model_name, num_epochs=25):
    save_model_dir = 'save_models/{}'.format(save_model_name)
    os.makedirs(save_model_dir, exist_ok=True)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    task = 'binary'
    
    if len(class_names) == 2:
        print('Task: Binary Class')
        task = 'binary'
    else:
        print('Task: Multi Class')
        task = 'multi'
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                experiment.train()
            else:
                model.eval()
                experiment.validate()

            running_loss = 0.0
            running_corrects = 0
            
            labels_all = []
            pred_all = []

            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                pred_all.extend(predict.item() for predict in preds)
                labels_all.extend(label.item() for label in labels)
                                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            metrics_dict = classification_report(y_true=labels_all, y_pred=pred_all, output_dict=True, zero_division=0)

            epoch_recall = metrics_dict['macro avg']['recall']
            epoch_precision = metrics_dict['macro avg']['precision']
            epoch_f1 = metrics_dict['macro avg']['f1-score']

            print('{} Loss: {:.4f} Acc: {:.4f} Recall: {:.4f} Precision: {:.4f} F1-score: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_recall, epoch_precision, epoch_f1))
            
            experiment.log_metric('{}_loss'.format(phase), epoch_loss, step=epoch)
            experiment.log_metric('{}_acc'.format(phase), epoch_acc, step=epoch)
            experiment.log_metric('{}_recall'.format(phase), epoch_recall, step=epoch)
            experiment.log_metric('{}_precision'.format(phase), epoch_precision, step=epoch)
            experiment.log_metric('{}_f1'.format(phase), epoch_f1, step=epoch)
            
            fig = plot_roc_fig(labels_all, pred_all, class_names, task)
            experiment.log_figure('ROC', fig, step=epoch)
            
            if phase == 'valid':
                confusion_matrix.compute_matrix(labels_all, pred_all)
                experiment.log_confusion_matrix(matrix=confusion_matrix,
                                                title='Confusion Matrix, Epoch #{}'.format(epoch + 1),
                                                file_name='confusion-matrix-{}.json'.format(epoch + 1)
                                                )
        
            # best modelの保存
            if phase == 'valid' and epoch_acc > best_acc:
                # if epoch_recall==1 and epoch_precision > best_precision:
                #     torch.save(model.state_dict(), 
                #                os.path.join(save_model_dir, save_model_name+'_{}_{}_recall_1.0.pkl'.format(epoch)))
                #     print('saving model epoch :{}'.format(epoch))
                #     recall_1_precision = epoch_precision
                best_precision = epoch_precision
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
        print()
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 
               os.path.join(save_model_dir, save_model_name+'_bs{}_lr{}_best.pkl'.format(hyper_params['batch_size'], hyper_params['learning_rate'])))
    experiment.log_model('best_model',
                          os.path.join(save_model_dir,save_model_name+'_bs{}_lr{}_best.pkl'.format(hyper_params['batch_size'], hyper_params['learning_rate'])))
    
    experiment.log_metric('best_val_acc', best_acc)
    
    print('-' * 10)
    print('Best val Acc: {:4f}, Precision: {:.4f}'.format(best_acc, best_precision))
    print('Fin')

    return model

# test画像で検証 -> 可視化（comet.mlへ転送）
def visualize_model_cometml(experiment, confusion_matrix, model, dataloaders, class_names, device):
    experiment.test()
    model.eval() 

    with torch.no_grad():
        correct = 0
        total = 0
           
        labels_all = []
        pred_all = []
           
        for idx, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            correct += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
            pred_all.extend(predict.item() for predict in preds)
            labels_all.extend(label.item() for label in labels)
            
        print('Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
        experiment.log_metric('test_acc', 100. * correct / total)

    confusion_matrix.compute_matrix(labels_all, pred_all)
    experiment.log_confusion_matrix(matrix=confusion_matrix,
                                    title='Test Confusion Matrix',
                                    file_name='test-confusion-matrix.json'
                                    )
