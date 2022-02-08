import collections
from collections import Counter
import argparse
import glob
import json
import numpy as np
import os
import pytorch_lightning as pl
import random
import string
import torch
import transformers
import pandas as pd
from num2words import num2words
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List
from rerank import *

inverted_actions_map = {
    'I_WALK': 'walk',
    'I_JUMP': 'jump',
    'I_LOOK': 'look',
    'I_TURN_RIGHT': 'turn right',
    'I_TURN_LEFT': 'turn left',
    'I_RUN': 'run'}


def make_scan_easy(target: str):
    for old, new in inverted_actions_map.items():
        target = target.replace(old, new)
    return target

def compute_exact_match(predicted_answer, correct_answer):
    predicted_answer = predicted_answer.strip().lower()
    correct_answer = correct_answer.strip().lower()
    return predicted_answer == correct_answer

class T5Finetuner(pl.LightningModule):

    def __init__(self, hparams2, train_dataloader, val_dataloader, test_dataloader):
        super(T5Finetuner, self).__init__()

        self.hparams2 = hparams2

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams2.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams2.model_name_or_path)


        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        
        if self.hparams2.restore_logging == 'None':
            self.metrics = pd.DataFrame({'accuracy':[], 'epoch':[]})
        else:
            self.metrics = pd.read_csv(self.hparams2.restore_logging, sep='\t')

    def prepare_batch(self, questions: List[str], answers: List[str]) -> List[str]:

        input_dict = self.tokenizer.batch_encode_plus(
            list(questions), padding=True, truncation=False, return_tensors='pt')

        assert input_dict['input_ids'].shape[1] < self.hparams2.max_seq_length
        
        labels = self.tokenizer.batch_encode_plus(
            list(answers), padding=True, truncation=False, return_tensors='pt')['input_ids']

        assert labels.shape[1] < self.hparams2.max_seq_length            

        input_ids = input_dict['input_ids'].to(self.model.device)
        attention_mask = input_dict['attention_mask'].to(self.model.device)
        labels = labels.to(self.model.device)
        return input_ids, attention_mask, labels

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_nb):
        questions, correct_answers, _ = batch

        # Log every power of two.
        if batch_nb & (batch_nb - 1) == 0:
            print(questions[0])
            print(correct_answers[0])

        input_ids, attention_mask, labels = self.prepare_batch(
            questions=questions, answers=correct_answers)

        loss = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels).loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        
        #self.inference_step(batch, batch_nb)

        
        return {'loss': loss, 'batch_loss': loss.cpu().detach().numpy().tolist(),
                'batch':batch}
    
 
    def inference_step(self, batch, batch_nb: int):
        questions, correct_answers, num_digits = batch

        input_ids, attention_mask, _ = self.prepare_batch(
            questions=questions, answers=correct_answers)

        
        batch_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_length=self.hparams2.max_seq_length)

        predicted_answers = [
            self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in batch_outputs]

        exact_matches = [
            compute_exact_match(predicted_answer=predicted_answer, correct_answer=correct_answer)
            for predicted_answer, correct_answer in zip(predicted_answers, correct_answers)]

        # Log every power of two.
        if batch_nb & (batch_nb - 1) == 0:
            print('\nQuestion:', questions[0])
            print('Correct:  ', correct_answers[0])
            print('Predicted:', predicted_answers[0].encode('utf-8'))
            print('Exact?', exact_matches[0])
        
        metrics = {'exact_matches': exact_matches, 
                   'predicted_answers':predicted_answers}
        return metrics

    def validation_step(self, batch, batch_nb):
        return {'batch':batch, 
                'out_model':self.inference_step(batch, batch_nb)}

    def test_step(self, batch, batch_nb):
        return {'batch':batch, 
                'out_model':self.inference_step(batch, batch_nb)}
                
    def compute_evaluations(self, exact_matches, correct_answers, predicted_answers, type_eval='val_'):

        exact_match = sum(exact_matches) / len(exact_matches)
        #create reports ...
        print(classification_report(correct_answers, predicted_answers))
        test_recall = None
        
        if self.hparams2.result_checkpoint != 'None':
            answers_test =  read_checkpoint(self.hparams2.result_checkpoint)
            test_recall = Counter(eval_modelo(answers_test,self))
            
        cm = confusion_matrix(correct_answers, predicted_answers)
        cmd = ConfusionMatrixDisplay(cm, display_labels=['true','false'])
        cmd.plot()
        plt.savefig(os.path.join(self.hparams2.output_dir, 'confusion_matrix.png'))
        plt.close()
        self.log(type_eval+'exact_match', exact_match, on_step=False, on_epoch=True, prog_bar=True)
        
        if test_recall != None:
            self.log(type_eval+'recall_rerank_true', test_recall[True], on_step=False, on_epoch=True, prog_bar=True)
            self.log(type_eval+'recall_rerank_false', test_recall[False], on_step=False, on_epoch=True, prog_bar=True)
            self.log('recall_rerank_true_percentagem', (test_recall[True]*100)/(test_recall[True]+test_recall[False]), on_step=False, on_epoch=True, prog_bar=True)
        
            self.metrics = self.metrics.append({'accuracy':str(exact_match), 'epoch':str(self.current_epoch), 
                                                'recall_rerank_true':test_recall[True],
                                                'recall_rerank_false':test_recall[False],
                                                'recall_rerank_true_percentagem':(test_recall[True]*100)/(test_recall[True]+test_recall[False])}, ignore_index = True)
                                            
            self.metrics.to_csv(os.path.join(self.hparams2.output_dir, 'metrics_seed_'+str(self.hparams2.seed)+'_test.tsv'),sep='\t', index= False)
              
    def validation_epoch_end(self, outputs):
    
        exact_matches = []
        predicted_answers = []
        correct_answers = []
        questions_all = []
        for x in outputs:
            exact_matches.extend(x['out_model']['exact_matches'])
            questions, correct_answer, _ = x['batch']

            questions_all.extend(questions)
            predicted_answers.extend(x['out_model']['predicted_answers'])
            correct_answers.extend(correct_answer)
            
        self.compute_evaluations(exact_matches, correct_answers, predicted_answers)
        return

    def test_epoch_end(self, outputs):
        exact_matches = []
        predicted_answers = []
        correct_answers = []
        questions_all = []
        for x in outputs:
            exact_matches.extend(x['out_model']['exact_matches'])
            questions, correct_answer, _ = x['batch']

            questions_all.extend(questions)
            predicted_answers.extend(x['out_model']['predicted_answers'])
            correct_answers.extend(correct_answer)

        exact_match = sum(exact_matches) / len(exact_matches)
        print('Exact match: ', exact_match)
        self.compute_evaluations(exact_matches, correct_answers, predicted_answers, type_eval='test_')       
        return 

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def get_optimizer(self):
        optimizer_name = self.hparams2.optimizer
        scheduler_name = self.hparams2.scheduler
        lr = self.hparams2.lr
        weight_decay = self.hparams2.weight_decay

        if optimizer_name.lower() == 'adafactor':
            optimizer = transformers.Adafactor
            optimizer = optimizer(self.parameters(), lr=lr, weight_decay=weight_decay, relative_step=False)
        else:
            optimizer = getattr(torch.optim, optimizer_name)

            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            optimizer = optimizer(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)

        print(f'=> Using {optimizer_name} optimizer')

        if scheduler_name == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams2.t_0, T_mult=self.hparams2.t_mult, eta_min=5e-7)
            print(f'=> Using CosineAnnealingWarmRestarts (T_0 = {self.hparams2.t_0}, T_mult = {self.hparams2.t_mult}, eta_min = 5e-7)')
        elif scheduler_name == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams2.gamma)
            print(f'=> Using ExponentialLR (gamma = {self.hparams2.gamma})')
        elif scheduler_name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams2.step_size, gamma=self.hparams2.gamma)
            print(f'=> Using StepLR (step_size = {self.hparams2.step_size}, gamma = {self.hparams2.gamma})')
        elif scheduler_name == 'WarmUp':
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams2.step_size)
            print(f'=> Using WarmUp (num_warmup_steps = {self.hparams2.step_size})')
        else:
            raise Exception(f'Scheduler not implemented: {scheduler_name}')

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return optimizer

class TsvDataset(Dataset):
    def __init__(self, path: str):
        self.examples = pd.read_csv(path, sep ='\t')
        self.question = self.examples['question'].to_list()
        self.hipotese = self.examples['hipotese'].to_list()
        self.classes = self.examples['class'].to_list()
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return [str('Instruction: '+self.question[idx]+' is equal: '+self.hipotese[idx]), str(self.classes[idx]), 0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evalute T5 on arithmetic problems.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--result_checkpoint', type=str, default='None')
    parser.add_argument('--restore_checkpoint', type=str, default='')
    parser.add_argument('--test_checkpoint', type=str, default='None')
    parser.add_argument('--restore_logging', type=str, default='None')
    parser.add_argument('--input_train', type=str, default=None)
    parser.add_argument('--input_val', type=str, default=None)
    parser.add_argument('--input_test', type=str, default=None)
    parser.add_argument("--seed", default=123, type=int, help="Seed.")
    parser.add_argument("--train_size", default=1000, type=int, help="Number of examples for training.")
    parser.add_argument("--val_size", default=1000, type=int, help="Number of examples for training.")
    parser.add_argument("--test_size", default=2000, type=int, help="Number of examples for testing.")
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length (in tokens).')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--scheduler', type=str, default='StepLR',
                        help='learning rate scheduler. Choose among (CosineAnnealingWarmRestarts, ExponentialLR and StepLR)')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma factor for ExponentialLR or StepLR')
    parser.add_argument('--step_size', type=int, default=2, help='period of learning rate decay (StepLR)')
    parser.add_argument('--t_0', type=int, default=2,
                        help='number of iterations for the first restart (CosineAnnealingWarmRestarts)')
    parser.add_argument('--t_mult', type=int, default=2,
                        help='a factor increases t_i after a restart (CosineAnnealingWarmRestarts)')
    parser.add_argument("--num_workers", default=4, type=int, help="Number of CPU workers for loading data.")

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()

    print('args', args)
    # print('unknown', unknown)

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    pl.seed_everything(args.seed)

    #datasets
    dataset_train = TsvDataset(args.input_train)

    dataset_val = TsvDataset(args.input_val)


    dataset_test = TsvDataset(args.input_test)

    #dataloaders
    train_dataloader = DataLoader(dataset_train, batch_size=args.train_batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    val_dataloader = DataLoader(dataset_val, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers)

    test_dataloader = DataLoader(dataset_test, batch_size=args.val_batch_size,
                                 shuffle=False, num_workers=args.num_workers)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir, filename=''+str(args.lr)+'-'+str(args.seed)+'-{epoch}-{recall_rerank_true_percentagem:.4f}',
        verbose=False, save_last=True, save_top_k=1, mode='max', monitor='recall_rerank_true_percentagem',
        save_weights_only=False, every_n_val_epochs=args.check_val_every_n_epoch)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=checkpoint_callback)

    if args.restore_checkpoint != '':
        print("Restore checkpoint:", args.restore_checkpoint)
        model = T5Finetuner.load_from_checkpoint(args.restore_checkpoint,
                                            hparams2=args,
                                            train_dataloader=train_dataloader,
                                            val_dataloader=val_dataloader,
                                            test_dataloader=test_dataloader)
    else:
        model = T5Finetuner(hparams2=args,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            test_dataloader=test_dataloader)

    #Test    
    if args.test_checkpoint != 'None':
        trainer.test(model)
    else:
        trainer.fit(model)
        trainer.test(model)


    print('Done!')
