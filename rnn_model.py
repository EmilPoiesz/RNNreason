import argparse
import torch
import os
import json

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from dataset import RNNLogicDataset

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) 
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])

        return out
    
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        
        return out

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        None
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        
        return out

    
def train(args, model, train_dataset, eval_dataset=None):
    
    args.train_batch_size = args.train_batch_size_per_GPU
    
    train_sampler = SequentialSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate_fn,
                              sampler=train_sampler, 
                              batch_size=args.train_batch_size)
    
    total_steps = len(train_loader) * args.num_epochs
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=args.num_warmup_steps,
                                               num_training_steps=total_steps)

    criterion = nn.BCEWithLogitsLoss().to(args.device)
    accuracy = BinaryAccuracy().to(args.device)
    precision = BinaryPrecision().to(args.device)
    recall = BinaryRecall().to(args.device)
    conf_matrix = BinaryConfusionMatrix().to(args.device)
    f1 = BinaryF1Score().to(args.device)
    
    print("**** Running training ****")
    train_loss = 0.0
    global_step = 0
    accs = 0.0
    precisions = 0.0
    recalls = 0.0
    f1_value = 0.0
    
    model.zero_grad()
    for epoch in range(args.num_epochs):
        
        depths_counter = [0, 0, 0, 0, 0, 0, 0]
        conf_by_depths = [[],[],[],[],[],[],[]]
        conf_ = []
        
        for i, batch in enumerate(train_loader):
            
            embeddings, labels, depths = batch
            embeddings = embeddings.to(args.device)
            labels = labels.to(args.device)
            depths = depths.to(args.device)
            
            model.train()
            outputs = model(embeddings)
            preds = outputs.squeeze()
            
            loss = criterion(preds, labels)
            loss.backward()
            train_loss += loss.item()
            
            grouped_by_depths_preds = {f"depth_{k}": preds[depths == float(k)] for k in range(7)}
            grouped_by_depths_labels = {f"depth_{k}": labels[depths == float(k)] for k in range(7)}
            
            for j in range(7):
                depths_counter[j] += len(grouped_by_depths_preds[f'depth_{j}'])
                conf_by_depths[j].append(conf_matrix(grouped_by_depths_preds[f'depth_{j}'],
                                                     grouped_by_depths_labels[f'depth_{j}']))

            accs += accuracy(preds, labels).item()
            precisions += precision(preds, labels).item()
            recalls += recall(preds, labels).item()
            f1_value += f1(preds, labels).item()
            conf_.append(conf_matrix(preds, labels))
            
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            
            if i % 100 == 0:
                print(f"----Batch {i}/{len(train_loader)} finished----  Loss: {loss.item()}")
                
        print(f"----Epoch {epoch+1}/{args.num_epochs} finished----")
        print(f"Train loss: {train_loss/global_step}")
        
        metrics = {"Epoch": epoch+1,
                   "loss": train_loss/global_step,
                   "accuracy": accs/global_step, 
                   "precision": precisions/global_step, 
                   "recall": recalls/global_step,
                   "f1": f1_value/global_step,
                   "confusion matrix": sum(conf_).tolist(),
                   "depths": depths_counter,
                   "confusion matrix by depth": [sum(conf).tolist() for conf in conf_by_depths]
                  }
        eval_metrics = evaluate(args, eval_dataset, model) 
        eval_metrics["Epoch"] = epoch+1
        
        if not os.path.isfile(f"{args.model_type}_{args.num_layers}_metrics.txt"):
            with open(f"{args.model_type}_{args.num_layers}_metrics.txt", 'a') as file: pass
        with open(f"{args.model_type}_{args.num_layers}_metrics.txt", 'a') as file:
            file.write(json.dumps(metrics))
            file.write('\n')
        if not os.path.isfile(f"{args.model_type}_{args.num_layers}_evalmetrics.txt"):
            with open(f"{args.model_type}_{args.num_layers}_evalmetrics.txt", 'a') as file: pass
        with open(f"{args.model_type}_{args.num_layers}_evalmetrics.txt", 'a') as file:
            file.write(json.dumps(eval_metrics))
            file.write('\n')
        
        
        # Saving model after each epoch
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(epoch))
        print("Saving model checkpoint to ", output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)    
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        print()


def evaluate(args, eval_dataset, model):
    
    args.eval_batch_size = args.eval_batch_size_per_GPU
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, 
                                 collate_fn=eval_dataset.collate_fn,
                                 sampler=eval_sampler, 
                                 batch_size=args.eval_batch_size)

    criterion = nn.BCEWithLogitsLoss().to(args.device)
    accuracy = BinaryAccuracy().to(args.device)
    precision = BinaryPrecision().to(args.device)
    recall = BinaryRecall().to(args.device)
    f1 = BinaryF1Score().to(args.device)
    conf_matrix = BinaryConfusionMatrix().to(args.device)
    
    print("**** Running evaluation ****")
    print()
    
    eval_loss = 0.0
    global_step = 0
    accs = 0.0
    precisions = 0.0
    recalls = 0.0
    f1_value = 0.0
    conf_ = []
    
    depths_counter = [0, 0, 0, 0, 0, 0, 0]
    conf_by_depths = [[],[],[],[],[],[],[]]
    
    for batch in eval_dataloader:
        model.eval()
        
        with torch.no_grad():
            embeddings, labels, depths = batch
            embeddings = embeddings.to(args.device)
            labels = labels.to(args.device)
            depths = depths.to(args.device)
            outputs = model(embeddings)
        preds = outputs.squeeze()
        
        grouped_by_depths_preds = {f"depth_{k}": preds[depths == float(k)] for k in range(7)}
        grouped_by_depths_labels = {f"depth_{k}": labels[depths == float(k)] for k in range(7)}

        for i in range(7):
            depths_counter[i] += len(grouped_by_depths_preds[f'depth_{i}'])
            conf_by_depths[i].append(conf_matrix(grouped_by_depths_preds[f'depth_{i}'], grouped_by_depths_labels[f'depth_{i}']))

        loss = criterion(preds, labels)
        eval_loss += loss.item()
        global_step += 1

        accs += accuracy(preds, labels).item()
        precisions += precision(preds, labels).item()
        recalls += recall(preds, labels).item()
        f1_value += f1(preds, labels).item()
        conf_.append(conf_matrix(preds, labels))
    
    metrics = {"Epoch": None,
               "loss": eval_loss/global_step,
               "accuracy": accs/global_step, 
               "precision": precisions/global_step, 
               "recall": recalls/global_step,
               "f1": f1_value/global_step,
               "confusion matrix": sum(conf_).tolist(),
               "depths": depths_counter,
               "confusion matrix by depth": [sum(conf).tolist() for conf in conf_by_depths]
              }
    return metrics 

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_type",
                       type=str,
                       required=True,
                       help="Model to train")
    

    parser.add_argument("--tokenizer_name", default="bert-base-uncased", type=str)
    parser.add_argument("--input_dimension", default=768, type=int)
    parser.add_argument("--hidden_dimension", default=768, type=int)
    parser.add_argument("--num_layers", default=1, type=int)

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--num_warmup_steps", default=0.1, type=float)
    
    parser.add_argument("--train_file_path", default="DATA/RP/prop_examples.balanced_by_backward.max_6.json_train", type=str)
    parser.add_argument("--val_file_path", default="DATA/RP/prop_examples.balanced_by_backward.max_6.json_val", type=str)
    parser.add_argument("--test_file_path", default="DATA/RP/prop_examples.balanced_by_backward.max_6.json_test", type=str)
    
    parser.add_argument("--train_batch_size_per_GPU", default=64, type=int) 
    parser.add_argument("--eval_batch_size_per_GPU", default=64, type=int)
    parser.add_argument("--num_epochs", default=20, type=int) 
    
    parser.add_argument("--keep_only_negative", action="store_true")
    parser.add_argument("--limit_example_num", default=-1, type=int)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--skip_long_examples", action="store_true")
    parser.add_argument("--ignore_fact", action="store_true")
    parser.add_argument("--ignore_both", action="store_true")
    parser.add_argument("--ignore_query", action="store_true")
    parser.add_argument("--shorten_input", action="store_false")
    parser.add_argument('--shrink_ratio', default=1, type=int)
    parser.add_argument('--use_autocast', action="store_true")
    parser.add_argument('--model_name_or_path', default="rnn", type=str)
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument("--custom_weight", default=None, type=str)
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        args.num_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        args.num_gpu = 0
    args.device = device
    
    args.model_type = args.model_type.lower()
    if "rnn" in args.model_type:
        model = RNNClassifier(args.input_dimension, args.hidden_dimension, args.num_layers)
    if "lstm" in args.model_type:
        model = LSTMClassifier(args.input_dimension, args.hidden_dimension, args.num_layers)
    if "gru" in args.model_type:
        model = GRUClassifier(args.input_dimension, args.hidden_dimension, args.num_layers)
    model.to(args.device)
    
    if args.do_train:
        args.output_dir = f"OUTPUT/RP/{args.model_type}_{args.num_layers}"
        train_dataset = RNNLogicDataset.initialze_from_file(args.train_file_path, args)
        eval_dataset  = RNNLogicDataset.initialze_from_file(args.val_file_path, args)
        train(args, model, train_dataset, eval_dataset)
        
    if args.do_eval:
        args.output_dir = f"OUTPUT/EVAL/{args.model_type}_{args.num_layers}"
        test_dataset  = RNNLogicDataset.initialze_from_file(args.test_file_path, args)
        if args.custom_weight is not None:
            model.load_state_dict(torch.load(args.custom_weight))
        model.eval()
        test_metrics = evaluate(args, test_dataset, model)
        if not os.path.isfile(f"{args.model_type}_{args.num_layers}_testmetrics.txt"):
            with open(f"{args.model_type}_{args.num_layers}_testmetrics.txt", 'a') as file: pass
        with open(f"{args.model_type}_{args.num_layers}_testmetrics.txt", 'a') as file:
            file.write(json.dumps(test_metrics))
            file.write('\n')
    
if __name__ == "__main__":
    main()