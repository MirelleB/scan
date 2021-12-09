import collections
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

from num2words import num2words
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
# from transformers import BigBirdForCausalLM, BigBirdConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import neptune.new as neptune

from typing import List

#neptune
run = neptune.init(project='mirelle/scan',
                   api_token ='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1MmI3M2JhMi03NDQ1LTQ1YmItOTFhMy1hZjRlZjRjYmMxZjUifQ==')

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


def sample_word(original_word: str, sample_probability: float, 
                min_chars: int = 2, max_chars: int = 4):
    if random.random() <= sample_probability:
        return ''.join(random.choices(
            string.ascii_lowercase,
            k=random.randint(min_chars, max_chars)))
    return original_word


def augment_scan(source: str, target: str, sample_probability: float,
                 min_chars: int = 2, max_chars: int = 4):
    text = f'{source}\t{target}'
    forbidden_words = set(['and', 'after', 'opposite', 'around', 'turn', 'twice', 'thrice'])
    for word in ['jump', 'run', 'walk', 'look', 'right', 'left']:
        new_word = 'and'
        while new_word in forbidden_words:
            new_word = sample_word(
                word, sample_probability=sample_probability, min_chars=min_chars, max_chars=max_chars)
        text = text.replace(word, new_word)
        forbidden_words.add(new_word)
        
    return text.split('\t')


def compute_exact_match(predicted_answer, correct_answer):
    predicted_answer = predicted_answer.strip().lower()
    correct_answer = correct_answer.strip().lower()
    return predicted_answer == correct_answer


def convert_to_base(num: int, base: int, numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    return ((num == 0) and numerals[0]) or (convert_to_base(num // base, base, numerals).lstrip(numerals[0]) + numerals[num % base])


def convert_to_character(number: str, separator: str, invert_number: bool, max_digits: int, position_chars=None):
    if max_digits > 0:
        signal = None
        if number[0] == '-':
            signal = '-'
            number = number[1:]
        number = (max_digits - len(number)) * '0' + number
        if signal:
            number = signal + number

    if position_chars:
        number = ''.join(
            [f'{digit}{position}' for digit, position in zip(number[::-1], position_chars[::-1])])
        number = number[::-1]

    if invert_number:
        number = number[::-1]
    return separator.join(number)


def convert_to_10based(number: str, invert_number: bool):
    signal = None
    if number[0] == '-':
        signal = '-'
        number = number[1:]

    output = []
    for i, digit in enumerate(number[::-1]):
        if i > 0:
            output.append('1' + i * '0')
        output.append(digit)

    if signal:
        output.append(signal)

    # The output is already inverted. If we want it to _not_ be inverted, then we invert it.
    if not invert_number:
        output = output[::-1]

    return ' '.join(output)


def convert_to_10ebased(number: str, split_type: str, invert_number: bool):

    signal = None
    if number[0] == '-':
        signal = '-'
        number = number[1:]

    output = []
    for i, digit in enumerate(number[::-1]):
        if split_type is None:
            output.append('10e' + str(i))
        elif split_type == 'underscore':
            output.append('10e' + '_'.join(str(i)))
        elif split_type == 'character':
            output.append(' '.join('D' + str(i) + 'E'))
        else:
            raise Exception(f'Wrong split_type: {split_type}')
        output.append(digit)

    if signal:
        output.append(signal)

    # The output is already inverted. If we want it to _not_ be inverted, then we invert it.
    if not invert_number:
        output = output[::-1]

    return ' '.join(output)


class T5Finetuner(pl.LightningModule):

    def __init__(self, hparams2, train_dataloader, val_dataloader, test_dataloader):
        super(T5Finetuner, self).__init__()

        self.hparams2 = hparams2

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams2.model_name_or_path)
        if self.hparams2.seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams2.model_name_or_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams2.model_name_or_path)
        

        if self.hparams2.orthography.endswith('_fixed'):
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['0']})

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self.step = 0

    def prepare_batch(self, questions: List[str], answers: List[str]) -> List[str]:

        input_dict = self.tokenizer.batch_encode_plus(
            list(questions), padding=True, truncation=False, return_tensors='pt')

        assert input_dict['input_ids'].shape[1] < self.hparams2.max_seq_length
        
        if self.hparams2.seq2seq:
            labels = self.tokenizer.batch_encode_plus(
                list(answers), padding=True, truncation=False, return_tensors='pt')['input_ids']

            assert labels.shape[1] < self.hparams2.max_seq_length            
        else:
            labels = torch.tensor(answers)

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
        
        #compute batch acc
        result_metrics = self.inference_step(batch, batch_nb)

        
        return {'loss': loss, 'out_model':result_metrics, 
                'batch_loss': loss.cpu().detach().numpy().tolist(),
                'batch':batch}

    def training_epoch_end(self, outputs):
        exact_matches = []
        loss_epoch = []
        predicted_answers = []
        correct_answers = []
        questions_all = []

        for x in outputs:
            
            exact_matches.extend(x['out_model']['exact_matches'])
            loss_epoch.append(x['batch_loss'])

            questions, correct_answer, _ = x['batch'] 

            predicted_answers.extend(x['out_model']['predicted_answers'])
            correct_answers.extend(correct_answer)
            questions_all.extend(questions)

        exact_match = sum(exact_matches) / len(exact_matches)
        compute_loss = sum(loss_epoch) / len(self._train_dataloader.dataset)

        self.log('train_accuracy', exact_match, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', compute_loss, on_step=False, on_epoch=True, prog_bar=True)

        #save file json
        out_file = {"seed":self.hparams2.seed,
                     "loss":compute_loss,
                     "accuracy": exact_match,
                     "questions":questions_all,
                     "correct_answers":correct_answers,
                     "predict": predicted_answers}

        file_name ='checkpoint_'+str(self.hparams2.seed)+'_epoch_'+str(self.step)+'_training.json' 
        with open( os.path.join(self.hparams2.output_dir, file_name), 'w') as f:
            json.dump(out_file, f)
           
        #update step
        self.step+=1
        #log neptune
        run['train/accuracy'].log(exact_match)
        run['train/loss'].log(compute_loss)
        return


    def inference_step(self, batch, batch_nb: int):
        questions, correct_answers, num_digits = batch

        input_ids, attention_mask, _ = self.prepare_batch(
            questions=questions, answers=correct_answers)

        if self.hparams2.seq2seq:
            batch_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_length=self.hparams2.max_seq_length)

            predicted_answers = [
                self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for output in batch_outputs]

        else:
            batch_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_answers = batch_outputs.logits.argmax(1)
            predicted_answers = list(map(str, predicted_answers.tolist()))
            correct_answers = list(map(str, correct_answers.tolist()))
            
        

        exact_matches = [
            compute_exact_match(predicted_answer=predicted_answer, correct_answer=correct_answer)
            for predicted_answer, correct_answer in zip(predicted_answers, correct_answers)]

        # Log every power of two.
        if batch_nb & (batch_nb - 1) == 0:
            print('\nQuestion:', questions[0])
            print('Correct:  ', correct_answers[0])
            print('Predicted:', predicted_answers[0].encode('utf-8'))
            print('Exact?', exact_matches[0])
        
        metrics = {'exact_matches': exact_matches, 'num_digits': num_digits.tolist(), 
                   'predicted_answers':predicted_answers}
        return metrics

    def validation_step(self, batch, batch_nb):
        return {'batch':batch, 
                'out_model':self.inference_step(batch, batch_nb)}

    def test_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)

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

        exact_match = sum(exact_matches) / len(exact_matches)

        self.log('val_exact_match', exact_match, on_step=False, on_epoch=True, prog_bar=True)
        
        #save file json
        print()
        out_file = {"seed":self.hparams2.seed,
                     "accuracy": exact_match,
                     "questions":questions_all,
                     "correct_answers":correct_answers,
                     "predict": predicted_answers}

        file_name ='checkpoint_'+str(self.hparams2.seed)+'_epoch_'+str(self.step) +'_test.json' 
        with open( os.path.join(self.hparams2.output_dir, file_name), 'w') as f:
            json.dump(out_file, f)
        # update step_val
        run['val/accuracy'].log(exact_match)
        return

    def test_epoch_end(self, outputs):
        exact_matches = collections.defaultdict(list)
        exact_match_all = []
        len_answers_all = []
        for batch in outputs:
            for num_digits, exact_match, len_answers in zip(batch['num_digits'], batch['exact_matches'], batch['len_real_answers']):
                exact_matches[num_digits].append(exact_match)
                exact_match_all.append(exact_match)
                len_answers_all.append(len_answers)
                

        for num_digits in sorted(list(exact_matches.keys())):
            exact_match = sum(exact_matches[num_digits]) / max(1, len(exact_matches[num_digits]))
            self.log(f'test_exact_match_{num_digits:02d}', exact_match, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'test_size_{num_digits:02d}', len(exact_matches[num_digits]), on_step=False, on_epoch=True, prog_bar=True)
            
        exact_match_all = sum(exact_match_all) / max(1, len(exact_match_all))
        avg_lean_anwers = sum(len_answers_all)/len(len_answers_all)
        
        # Convert in numpy array
        np_exact_match_all = np.array(exact_match_all)
        np_len_answers_all = np.array(len_answers_all)
        
        #get index of "true" values in exact_match_all
        idx_true = np.where(np_exact_match_all==True)[0]

        if len(idx_true) > 0:
            #no empty
            avg_len_correct_match = sum(np_len_answers_all[idx_true])/ len(np_len_answers_all[idx_true])
        else:
            avg_len_correct_match = 0

        self.log('test_exact_match_all', exact_match_all, on_step=False, on_epoch=True, prog_bar=True)
        self.log('avg_len_test examples', avg_lean_anwers, on_step=False, on_epoch=True, prog_bar=True)
        self.log('avg_len_correct_test examples', avg_len_correct_match, on_step=False, on_epoch=True, prog_bar=True)

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


class MyDataset(Dataset):
    def __init__(self, n_examples: int, min_digits: int, max_digits: int, operation: str,
                 orthography: str, base_number: int, invert_question: bool, invert_answer: bool,
                 balance: bool, seq2seq: bool):

        self.operation = operation
        self.orthography = orthography
        self.invert_answer = invert_answer
        self.invert_question = invert_question
        self.base_number = base_number
        self.max_digits = max_digits
        self.seq2seq = seq2seq

        if self.base_number != 10:
            assert self.orthography != 'words', 'Cannot convert to words when base is different than 10.'
            assert self.operation == 'addition', 'Cannot perform {self.operation} when base is different than 10.'

        if self.invert_question or self.invert_answer:
            assert self.orthography != 'words', 'Cannot invert number when ortography = "words".'

        if balance:
            self.examples = []
            for _ in range(n_examples):
                example = []
                max_digits_1 = random.randint(min_digits, max_digits)
                max_digits_2 = random.randint(min_digits, max_digits_1)
                temp = [max_digits_1, max_digits_2]
                random.shuffle(temp)
                for max_digits_i in temp:
                    min_number = int((max_digits_i - 1) * '9') + 1
                    max_number = int(max_digits_i * '9')
                    example.append(random.randint(min_number, max_number))
                self.examples.append(example)
        else:
            self.examples = [
                (random.randint(0, int(max_digits * '9')), random.randint(0, int(max_digits * '9')))
                for _ in range(n_examples)]

        if self.operation == 'index':
            self.examples = [
                (first_term, np.random.randint(low=0, high=len(str(first_term)) - 1, size=len(str(second_term))).tolist())
                for first_term, second_term in self.examples]

        print(sum([first_term > second_term for first_term, second_term in self.examples]) / len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        first_term, second_term = self.examples[idx]

        if self.operation == 'addition':
            operation_term = 'plus'
            result = first_term + second_term
        elif self.operation == 'subtraction':
            operation_term = 'minus'
            result = first_term - second_term
        elif self.operation == 'multiplication':
            operation_term = 'times'
            result = first_term * second_term
        elif self.operation == 'division':
            operation_term = 'divided by'
            result = first_term // second_term
        elif self.operation == 'copy_first':
            result = first_term
        elif self.operation == 'index':
            result = [str(first_term)[idx] for idx in second_term]
        elif self.operation == 'bigger':
            operation_term = 'bigger than'
            result = int(first_term > second_term)
        else:
            raise Exception(f'Invalid operation: {self.operation}')

        num_digits = max(len(str(first_term)), len(str(second_term)))

        position_chars = set()
        while len(position_chars) < num_digits:
            position_chars.add(''.join(random.choices(string.ascii_lowercase, k=2)))
        position_chars = list(position_chars)
        if len(str(result)) > num_digits:
            # replace the last random position token by a constant one ("#"), so the model can
            # learn which positional token to generate when there is a carry for the most 
            # significant digit.
            position_chars = ['#'] + position_chars

        first_term = self.convert_number(first_term, position_chars, invert_number=self.invert_question)
        second_term = self.convert_number(second_term, position_chars, invert_number=self.invert_question)
        if self.seq2seq:
            answer = self.convert_number(result, position_chars, invert_number=self.invert_answer)
        else:
            answer = result

        
        if self.operation == 'copy_first':
            return f'Copy {first_term}', answer, num_digits
        elif self.operation == 'index':
            return f'Number {first_term} Index: {second_term}', answer, num_digits
        elif self.operation == 'bigger':
            return f'Is {first_term} {operation_term} {second_term}?', answer, num_digits
        else:
            return f'What is {first_term} {operation_term} {second_term}?', answer, num_digits

    def convert_number(self, number: str, position_chars, invert_number: bool):
        # number can be int or List

        if isinstance(number, int):
            number = str(number)
        elif isinstance(number, list):
            number = [str(item) for item in number]

        if self.base_number != 10:
            number = convert_to_base(num=int(number), base=self.base_number)

        if self.orthography == 'decimal':
            return convert_to_character(
                number=number, separator='', invert_number=invert_number,
                max_digits=-1)
        if self.orthography == 'decimal_random':
            return convert_to_character(
                number=number, separator='', invert_number=invert_number,
                max_digits=-1, position_chars=position_chars)
        elif self.orthography == 'character':
            return convert_to_character(
                number=number, separator=' ', invert_number=invert_number,
                max_digits=-1)
        elif self.orthography == 'character_random':
            return convert_to_character(
                number=number, separator=' ', invert_number=invert_number,
                max_digits=-1, position_chars=position_chars)
        elif self.orthography == 'character_fixed':
            return convert_to_character(
                number=number, separator=' ', invert_number=invert_number,
                max_digits=self.max_digits)
        elif self.orthography == 'underscore':
            return convert_to_character(
                number=number, separator='_', invert_number=invert_number,
                max_digits=-1)
        elif self.orthography == 'underscore_fixed':
            return convert_to_character(
                number=number, separator='_', invert_number=invert_number,
                max_digits=self.max_digits)
        elif self.orthography == 'words':
            return num2words(int(number))
        elif self.orthography == '10based':
            return convert_to_10based(number, invert_number=invert_number)
        elif self.orthography == '10ebased':
            return convert_to_10ebased(
                number, split_type=None, invert_number=invert_number)
        elif self.orthography == '10ebased_underscore':
            return convert_to_10ebased(
                number, split_type='underscore', invert_number=invert_number)
        elif self.orthography == '10ebased_character':
            return convert_to_10ebased(
                number, split_type='character', invert_number=invert_number)
        else:
            raise Exception(f'Wrong orthography: {self.orthography}')


class TsvDataset(Dataset):
    def __init__(self, path: str, easy_scan: bool, augment_scan: float = 0,
                 aug_min_chars: int = 2, aug_max_chars: int = 4):
        self.examples = [line.rstrip().split('\t') for line in open(path)]
        self.augment_scan = augment_scan
        self.aug_min_chars = aug_min_chars
        self.aug_max_chars = aug_max_chars

        if easy_scan:
            self.examples = [[source, make_scan_easy(target)] for source, target in self.examples]
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        source, target = self.examples[idx]

        if self.augment_scan > 0:
            if random.random() <= self.augment_scan:
                source, target = augment_scan(
                    source=source, target=target, sample_probability=self.augment_scan,
                    min_chars=self.aug_min_chars, max_chars=self.aug_max_chars)
        return [source, target, 0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evalute T5 on arithmetic problems.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--seq2seq', action='store_true')
    parser.add_argument('--input_train', type=str, default=None)
    parser.add_argument('--input_val', type=str, default=None)
    parser.add_argument('--input_test', type=str, default=None)
    parser.add_argument('--operation', type=str, required=True)
    parser.add_argument('--orthography', type=str, required=True)
    parser.add_argument('--invert_question', action='store_true')
    parser.add_argument('--invert_answer', action='store_true')
    parser.add_argument('--balance_train', action='store_true')
    parser.add_argument('--balance_val', action='store_true')
    parser.add_argument('--balance_test', action='store_true')
    parser.add_argument('--min_digits_train', type=int, default=2)
    parser.add_argument('--min_digits_test', type=int, default=2)
    parser.add_argument('--max_digits_train', type=int, required=True)
    parser.add_argument('--max_digits_test', type=int, required=True)
    parser.add_argument('--base_number', type=int, default=10)
    parser.add_argument('--easy_scan', action='store_true')
    parser.add_argument('--augment_scan', type=float, default=0, help="Probability of augmenting SCAN examples. Only applied to training.")
    parser.add_argument('--scan_aug_min_chars', type=int, default=2, help="Minimum number of chars to be samples when augmenting SCAN.")
    parser.add_argument('--scan_aug_max_chars', type=int, default=4, help="Maximum number of chars to be samples when augmenting SCAN.")
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

    if args.input_train:
        dataset_train = TsvDataset(args.input_train, easy_scan=args.easy_scan, augment_scan=args.augment_scan,
                                   aug_min_chars=args.scan_aug_min_chars, aug_max_chars=args.scan_aug_max_chars,)
    else:
        dataset_train = MyDataset(n_examples=args.train_size, min_digits=args.min_digits_train,
                                  max_digits=args.max_digits_train,
                                  operation=args.operation, orthography=args.orthography,
                                  base_number=args.base_number, invert_question=args.invert_question,
                                  invert_answer=args.invert_answer, balance=args.balance_train,
                                  seq2seq=args.seq2seq)

    if args.input_val:
        dataset_val = TsvDataset(args.input_val, easy_scan=args.easy_scan)
    else:
        dataset_val = MyDataset(n_examples=args.val_size, min_digits=args.min_digits_train,
                                max_digits=args.max_digits_train,
                                operation=args.operation, orthography=args.orthography,
                                base_number=args.base_number, invert_question=args.invert_question,
                                invert_answer=args.invert_answer, balance=args.balance_val,
                                seq2seq=args.seq2seq)

    if args.input_test:
        dataset_test = TsvDataset(args.input_test, easy_scan=args.easy_scan)
    else:
        dataset_test = MyDataset(n_examples=args.test_size,
                                 min_digits=args.min_digits_test,
                                 max_digits=args.max_digits_test,
                                 operation=args.operation, orthography=args.orthography,
                                 base_number=args.base_number, invert_question=args.invert_question,
                                 invert_answer=args.invert_answer, balance=args.balance_test,
                                 seq2seq=args.seq2seq)

    train_dataloader = DataLoader(dataset_train, batch_size=args.train_batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    val_dataloader = DataLoader(dataset_val, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers)

    test_dataloader = DataLoader(dataset_test, batch_size=args.val_batch_size,
                                 shuffle=False, num_workers=args.num_workers)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir, filename='{epoch}-{val_exact_match:.4f}',
        verbose=False, save_last=False, save_top_k=1, mode='max', monitor='val_exact_match',
        save_weights_only=False, every_n_val_epochs=args.check_val_every_n_epoch)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=checkpoint_callback)

    model = T5Finetuner(hparams2=args,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        test_dataloader=test_dataloader)

    trainer.fit(model)

    checkpoint_path = glob.glob(os.path.join(args.output_dir, '*.ckpt'))[0]
    model = T5Finetuner.load_from_checkpoint(checkpoint_path,
                                             hparams2=args,
                                             train_dataloader=train_dataloader,
                                             val_dataloader=val_dataloader,
                                             test_dataloader=test_dataloader)

    results = trainer.test(model)

    output = {'seed': args.seed,
              'max_digits_train': args.max_digits_train,
              'max_digits_test': args.max_digits_test,
              'test_exact_match': results[0]['test_exact_match_all']}

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as fout:
        json.dump(output, fout)

    print('Done!')