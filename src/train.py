# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/usr/bin/env python

import os
import sys
import json
import time
import torch
import submitit
import argparse
import numpy as np
import pandas as pd
import random
import models
from prepare_datasets import get_loaders
from tqdm import tqdm
import gc
from datetime import datetime



def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    


def parse_args():
    parser = argparse.ArgumentParser(description='Balancing baselines')
    parser = argparse.ArgumentParser(description='Balancing baselines')
    parser.add_argument(
        "--seed",
         type=int, 
         default=0, 
         help="A seed for reproducible training."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="civilcomments",
        help="The name of the dataset to use. Options: waterbirds, celeba, multinli, civilcomments."
    )
    parser.add_argument(
        "--method", 
        type=str, 
        default="erm", 
        choices=["erm", "tapt", "suby", "subg", "rwy", "rwg", "dro", "jtt"],
        help="The method to use to train the models",
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="../../dataset/civilcomments/",
        help="The path to the dataset directory."
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=10, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-4, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the ERM training dataloader.",
    )
    parser.add_argument(
        "--grad_acc",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience value for early stopping",
    )
    parser.add_argument(
        "--T", 
        type=int,
         default=2,
         help="Number of epochs to train the first pass of JTT"
    )
    parser.add_argument(
        "--up", 
        type=int, 
        default=20,
        help="Upweighing parameter for JTT."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='models/', 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--result_dir", 
        type=str, 
        default='results/', 
        help="Where to store the prediction results."
    )
    return vars(parser.parse_args())



def save_results(results_dict, arguments, params_str):
    for data_type in results_dict:
        preds, gold, group_labels = results_dict[data_type]['preds'], results_dict[data_type]['gold'], results_dict[data_type]['group_labels']
        df = pd.DataFrame(list(zip(preds, gold, group_labels)), columns=['preds', 'gold', 'group_labels'])               
                   
        if 'subsamples' in arguments['data_path']:
            result_dir = os.path.join(arguments['result_dir'], f"{arguments['dataset']}-subsamples/{data_type}")
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            file_name = os.path.join(result_dir, f"{arguments['dataset']}_subsamples_{data_type}_{params_str}best.csv")
        else:
            result_dir = os.path.join(arguments['result_dir'], f"{arguments['dataset']}/{data_type}")
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            file_name = os.path.join(result_dir, f"{arguments['dataset']}_{data_type}_{params_str}best.csv")
        if os.path.exists(file_name):
            os.remove(file_name)     
        df.to_csv(file_name, index=False)
        
        
        
def run_experiment(args):
    
    print("This process has the PID: ", os.getpid())
    print("\nArguments: \n")
    
    if args['dataset'] in ['multinli', 'civilcomments']:
        args['model_type'] = 'bert'
    else:
        args['model_type'] = 'resnet50'
        
    print("\nArguments:\n")
    for key, val in args.items():
        print(f"{key}:\t{val}")
        
    params_str = ""
    params = ['model_type', 'method', 'batch_size', 'learning_rate', 'weight_decay', 'grad_acc', 'seed']
    for param in params:
        if param in ['model_type', 'method']:
            params_str += f"{args[param]}_"
        else:
            params_str += f"{param}_{args[param]}_"
        
    start_time = time.time()
    set_random_seed(args["seed"])
    
    loaders = get_loaders(args["data_path"], args["dataset"], args["batch_size"], args["method"])
    for data_type, data_loader in loaders.items():
        print(f"{data_type} size: {len(data_loader.dataset)}\t{data_type} dataloader size: {len(data_loader)}")
            
            
    model = {
        "erm": models.ERM,
        "suby": models.ERM,
        "subg": models.ERM,
        "rwy": models.ERM,
        "rwg": models.ERM,
        "dro": models.GroupDRO,
        "jtt": models.JTT
    }[args["method"]](args, loaders["tr"])

    last_epoch = 0
    best_selec_val = float('-inf')
    patience = 0 
    
    if 'subsamples' in args['data_path']:
        model_dir = os.path.join(args['output_dir'], f"{args['dataset']}-subsamples/")
        model_config_dir = os.path.join(f"{args['output_dir']}_params", f"{args['dataset']}-subsamples/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(model_config_dir):
            os.makedirs(model_config_dir)
        best_model_ckpt = os.path.join(model_dir, f"{args['dataset']}_subsamples_{params_str}best.pt")
        best_model_config_file = os.path.join(model_config_dir, f"{args['dataset']}_subsamples_{params_str}best.json")
    else:
        model_dir = os.path.join(args['output_dir'], f"{args['dataset']}/")
        model_config_dir = os.path.join(f"{args['output_dir']}_params", f"{args['dataset']}/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(model_config_dir):
            os.makedirs(model_config_dir)
        best_model_ckpt = os.path.join(model_dir, f"{args['dataset']}_{params_str}best.pt")
        best_model_config_file = os.path.join(model_config_dir, f"{args['dataset']}_{params_str}best.json")
    
    print(f"\nbest model ckpt: {best_model_ckpt}")
    print(f"\nbest model config file: {best_model_config_file}")
    best_model_config = {key: val for key, val in args.items()}

    for epoch in range(last_epoch, args["num_epochs"]):
        start_time = time.time()
        
        if epoch == args["T"] + 1 and args["method"] == "jtt":
            loaders = get_loaders(
                args["data_path"],
                args["dataset"],
                args["batch_size"],
                args["method"],
                model.weights.tolist())

        train_loss = 0.0
        for index, (i, x, y, g) in enumerate(tqdm(loaders["tr"], desc="Training Iteration")):
            loss = model.update(i, x, y, g, epoch)
            train_loss += loss

        result = {
            "args": args, 
            "epoch": epoch, 
            "time": time.time() - start_time,
            "tr_loss": train_loss/len(loaders["tr"])
        }
        
        preds_gold = dict()
        for loader_name, loader in loaders.items():
            if loader_name != 'tr':
                avg_acc, group_accs, inference_loss, preds, gold, group_labels = model.accuracy(loader, loader_name)
                result[f"{loader_name}_loss"] = inference_loss
            else:
                avg_acc, group_accs, preds, gold, group_labels = model.accuracy(loader, loader_name)
                
            result["avg_acc_" + loader_name] = avg_acc
            result["group_wise_acc_" + loader_name] = group_accs
            result["std_" + loader_name] = np.std(group_accs)
            result["min_acc_" + loader_name] = min(group_accs)
            result["group_ordering_" + loader_name] = np.argsort(group_accs)
            
            preds_gold[loader_name] = {
                'preds': preds,
                'gold': gold,
                'group_labels': group_labels
            }

        selec_value = {
            "min_acc_va": result["min_acc_va"],
            "avg_acc_va": result["avg_acc_va"],
        }
        
        if selec_value['min_acc_va'] >= best_selec_val:
            patience = 0
            model.best_selec_val = selec_value['min_acc_va']
            best_selec_val = selec_value['min_acc_va']
            print(f"saving best model and results at epoch: {epoch}")
            
            if os.path.exists(best_model_ckpt):
                os.remove(best_model_ckpt)
            
            best_model_config['epoch'] = epoch
            best_model_config['time'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            json_object = json.dumps(best_model_config, indent=4)
            with open(best_model_config_file, "w") as outfile:
                outfile.write(json_object)
            outfile.close()
            del json_object
            
            model.save(best_model_ckpt)
            save_results(preds_gold, args, params_str)
            
        else:
            patience += 1
            print(f"\npatience: {patience}")

        print(f"\n\nEpoch {epoch} results: \n")
        for key, value in result.items():
            print(f"{key}:\t{value}")
            
        if patience == args['early_stopping_patience']:
            print("\n\nStopping training as min_acc_va has not increased for 10 consecutive epochs...\n\n")
            break

        del loader
        del result
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    args["pid"] = os.getpid()
    run_experiment(args)