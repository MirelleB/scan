from typing import List, Union, Optional, Mapping, Any, Tuple, Iterable
from copy import deepcopy
import torch
import json
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

TokenizerReturnType = Mapping[str, Union[torch.Tensor, List[int],
                                         List[List[int]],
                                         List[List[str]]]]

from transformers import PreTrainedModel

DecodedOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

def read_checkpoint(path_file):
    #Verificar se existe
    with open(path_file, "r") as json_file:
        return json.load(json_file)
        

class Query:
    """Class representing a query.
    A query contains the query text itself and potentially other metadata.
    Parameters
    ----------
    text : str
        The query text.
    id : Optional[str]
        The query id.
    """

    def __init__(self, text: str, id: Optional[str] = None):
        self.text = text
        self.id = id


class Text:
    """Class representing a text to be reranked.
    A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or
    even a short phrase.
    Parameters
    ----------
    text : str
        The text to be reranked.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    score : Optional[float]
        The score of the text. For example, the score might be the BM25 score
        from an initial retrieval stage.
    title : Optional[str]
        The text's title.
    """

    def __init__(self,
                 text: str,
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0,
                 title: Optional[str] = None):
        self.text = text
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.title = title
        
@dataclass
class QueryDocumentBatch:
    query: Query
    documents: List[Text]
    output: Optional[TokenizerReturnType] = None

    def __len__(self):
        return len(self.documents)


def greedy_decode(model: PreTrainedModel,
                  input_ids: torch.Tensor,
                  length: int,
                  attention_mask: torch.Tensor = None,
                  return_last_logits: bool = True) -> DecodedOutput:

    decode_ids = torch.full((input_ids.size(0), 1),
                            model.config.decoder_start_token_id,
                            dtype=torch.long).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True)
        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat([decode_ids,
                                next_token_logits.max(1)[1].unsqueeze(-1)],
                               dim=-1)
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids

class TokenizerEncodeMixin:
    tokenizer: PreTrainedTokenizer = None
    tokenizer_kwargs = None

    def encode(self, strings: List[str]) -> TokenizerReturnType:
        assert self.tokenizer and self.tokenizer_kwargs is not None, \
                'mixin used improperly'
        ret = self.tokenizer.batch_encode_plus(strings,
                                               **self.tokenizer_kwargs)
        ret['tokens'] = list(map(self.tokenizer.tokenize, strings))
        return ret


class QueryDocumentBatchTokenizer(TokenizerEncodeMixin):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int,
                 pattern: str = '{query} {document}',
                 **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer_kwargs = tokenizer_kwargs
        self.pattern = pattern



    def traverse_query_document(self,
            batch_input: QueryDocumentBatch) -> Iterable[QueryDocumentBatch]:
        query = batch_input.query
        for batch_idx in range(0, len(batch_input), self.batch_size):
            docs = batch_input.documents[batch_idx:batch_idx + self.batch_size]
            outputs = self.encode([self.pattern.format(
                                        query=query.text,
                                        document=doc.text) for doc in docs])
            yield QueryDocumentBatch(query, docs, outputs)

def rescore(query: Query, texts: List[Text],tokenizer, model, token_false_id, token_true_id) -> List[Text]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    texts = deepcopy(texts)
    batch_input = QueryDocumentBatch(query=query, documents=texts)
    for batch in tokenizer.traverse_query_document(batch_input):
        with torch.cuda.amp.autocast(enabled=False):
            input_ids = torch.LongTensor(batch.output['input_ids']).to('cpu')
            attn_mask = torch.Tensor(batch.output['attention_mask']).to('cpu')
            input_ids= input_ids.to(device)
            attn_mask = attn_mask.to(device)
            _, batch_scores = greedy_decode(model,
                                            input_ids,
                                            length=1,
                                            attention_mask=attn_mask,
                                            return_last_logits=True)

            batch_scores = batch_scores[:, [token_false_id, token_true_id]]
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            batch_log_probs = batch_scores[:, 1].tolist()
        for doc, score in zip(batch.documents, batch_log_probs):
            doc.score = score
    return texts


def sort_rerank(texts: List[Text]) -> List[Text]:
      """Sorts a list of texts
      """
      return sorted(texts, key=lambda x: x.score, reverse=True)

def eval_modelo(answers, model):
  token_id_false = [model.tokenizer('false', return_tensors="pt").input_ids[0].numpy()[0]]
  token_id_true = [model.tokenizer('true', return_tensors="pt").input_ids[0].numpy()[0]]
  
  recall = []
  for idx, answer in enumerate(answers):
    all_predicts = set(answer['preditions'])
    query = Query(text = answer['question'])
    texts = [Text(text=text) for text in all_predicts]
    tokenizer = QueryDocumentBatchTokenizer(tokenizer = model.tokenizer, batch_size=1)
    texts_scores_map = sort_rerank(rescore(query = query, texts = texts, tokenizer =  tokenizer, 
                                    model=model.model, token_false_id = token_id_false, 
                                    token_true_id = token_id_true))
    #compute recall
    recall.append(texts_scores_map[0].text == answer['answer'])
  return recall