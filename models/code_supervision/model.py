import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.clinical_bert.model import ClinicalBertWrapper, EncoderSentences
from models.clusterer.model import Clusterer
from utils import traceback_attention as ta, entropy, set_dropout, set_require_grad, get_code_counts


def tensor_to_none(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor) and t.dim() == 1 and t.size(0) == 0:
        return None
    return t


def abstract_loss_func(total_num_codes, scores, codes, num_codes, attention, traceback_attention, context_vec, article_sentences_lengths, clustering, labels, attention_sparsity=False, traceback_attention_sparsity=False, gamma=1):
    b, nq, _, _ = attention.shape
    if scores.dim() == 3:
        # scores: (batch, nq, total_num_codes) -> collapse over nq
        scores = scores.mean(dim=1)  # now (batch, total_num_codes)

    # Align dimensions if scores has more codes than labels
    if scores.shape[1] > labels.shape[1]:
        scores = scores[:, :labels.shape[1]]
    elif scores.shape[1] < labels.shape[1]:
        raise ValueError(f"Mismatch: scores shape {scores.shape} < labels shape {labels.shape}")

    if scores.shape != labels.shape:
        raise ValueError(f"Mismatch: scores shape {scores.shape} != labels shape {labels.shape}")


    positive_labels = labels.sum()
    negative_labels = labels.numel() - positive_labels
    pos_weight = negative_labels / positive_labels if positive_labels > 0 else torch.tensor(1.0, device=scores.device)

    losses = F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=pos_weight, reduction='none')
    loss = losses.mean() * b

    if attention_sparsity:
        attention_flat = attention.view(b, nq, -1).mean(dim=1)
        loss += gamma * entropy(attention_flat).mean() * b

    if traceback_attention_sparsity:
        traceback_flat = traceback_attention.view(b, nq, -1).mean(dim=1)
        loss += gamma * entropy(traceback_flat).mean() * b

    return loss


def loss_func_creator(attention_sparsity=False, traceback_attention_sparsity=False, gamma=1):
    def loss_func_wrapper(total_num_codes, scores, codes, num_codes, attention, traceback_attention, context_vec, article_sentences_lengths, clustering, labels):
        return abstract_loss_func(
            total_num_codes, scores, codes, num_codes, attention, traceback_attention, context_vec, article_sentences_lengths, clustering, labels,
            attention_sparsity=attention_sparsity,
            traceback_attention_sparsity=traceback_attention_sparsity,
            gamma=gamma
        )
    return loss_func_wrapper


def statistics_func(total_num_codes, scores, codes, num_codes, attention, traceback_attention, context_vec, article_sentences_lengths, clustering, labels):
    b, nq, ns, nt = attention.shape
    if scores.dim() == 3:
        scores = scores.mean(dim=1)  # (batch, total_num_codes)

    # Align score dimensions to match labels
    if scores.shape[1] < labels.shape[1]:
        pad = labels.shape[1] - scores.shape[1]
        scores = F.pad(scores, (0, pad))
    elif scores.shape[1] > labels.shape[1]:
        scores = scores[:, :labels.shape[1]]

    if codes is not None and codes.shape[1] != scores.shape[1]:
        codes = codes[:, :scores.shape[1]]

    if context_vec is not None and context_vec.shape[1] != scores.shape[1]:
        context_vec = context_vec[:, :scores.shape[1]]

    code_mask = (torch.arange(labels.size(1), device=labels.device) < num_codes.unsqueeze(1))
    positives = get_code_counts(total_num_codes, codes, code_mask, (scores > 0))
    true_positives = get_code_counts(total_num_codes, codes, code_mask, ((scores > 0) & (labels == 1)))
    relevants = get_code_counts(total_num_codes, codes, code_mask, labels)

    return {
        'positives': positives,
        'true_positives': true_positives,
        'relevants': relevants,
        'total_predicted': code_mask.sum(),
        'accuracy_sum': ((scores[code_mask] > 0).long() == labels[code_mask]).sum().float() * b / code_mask.sum(),
        'attention_entropy': entropy(attention.view(b, nq, ns * nt).mean(dim=1)).mean() * b,
        'traceback_attention_entropy': entropy(traceback_attention.view(b, nq, ns * nt).mean(dim=1)).mean() * b
    }





class Model(nn.Module):
    def __init__(self, outdim=64, total_num_codes=19278, sentences_per_checkpoint=10, device1='cpu', device2='cpu', freeze_bert=True, code_embedding_type_params=set([]), concatenate_code_embedding=False, dropout=.15, cluster=False):
        super(Model, self).__init__()
        self.clinical_bert_sentences = EncoderSentences(ClinicalBertWrapper, embedding_dim=outdim, truncate_tokens=50, truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device1)
        if freeze_bert:
            self.freeze_bert()
        else:
            self.unfreeze_bert(dropout=dropout)
        self.code_embedding_type_params = code_embedding_type_params
        num_code_embedding_types = len(code_embedding_type_params)
        self.code_embeddings = nn.Embedding(code_embedding_type_params['codes'][0], outdim) if 'codes' in code_embedding_type_params.keys() else None
        self.linearized_code_transformer = EncoderSentences(lambda : LinearizedCodesTransformer(total_num_codes), embedding_dim=outdim, truncate_tokens=50, truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device2) if 'linearized_codes' in code_embedding_type_params.keys() else None
        self.attention = nn.MultiheadAttention(outdim, 1)
        self.concatenate_code_embedding = concatenate_code_embedding
        self.linear = nn.Linear(outdim, total_num_codes)  # Output all codes
        self.linear2 = nn.Linear(outdim * num_code_embedding_types, outdim) if num_code_embedding_types > 1 else None
        self.linear3 = nn.Linear(2 * outdim, outdim) if concatenate_code_embedding else None
        self.device1 = device1
        self.device2 = device2
        self.cluster = cluster
        self.clusterer = Clusterer() if cluster else None

    def freeze_bert(self):
        set_dropout(self.clinical_bert_sentences, 0)
        set_require_grad(self.clinical_bert_sentences, False)

    def unfreeze_bert(self, dropout=.15):
        set_dropout(self.clinical_bert_sentences, dropout)
        set_require_grad(self.clinical_bert_sentences, True)

    def correct_devices(self):
        self.clinical_bert_sentences.correct_devices()
        if self.code_embeddings is not None:
            self.code_embeddings.to(self.device2)
        self.attention.to(self.device2)
        self.linear.to(self.device2)
        if self.linear2 is not None:
            self.linear2.to(self.device2)
        if self.cluster:
            self.clusterer.to(self.device2)
        if self.concatenate_code_embedding:
            self.linear3.to(self.device2)

    def forward(self, article_sentences, article_sentences_lengths, num_codes, codes=None, code_description=None, code_description_length=None, linearized_codes=None, linearized_codes_lengths=None, linearized_descriptions=None, linearized_descriptions_lengths=None):
        scores, attention, traceback_attention, context_vec = self.inner_forward(
            article_sentences,
            article_sentences_lengths,
            num_codes,
            codes,
            code_description,
            code_description_length,
            linearized_codes,
            linearized_codes_lengths,
            linearized_descriptions,
            linearized_descriptions_lengths,
            *self.parameters()
        )
        if self.cluster:
            clustering = self.clusterer(article_sentences, article_sentences_lengths, attention, num_codes)
        else:
            clustering = None
        return_dict = dict(
            scores=scores,
            num_codes=num_codes,
            attention=attention,
            traceback_attention=traceback_attention,
            article_sentences_lengths=article_sentences_lengths,
            clustering=clustering,
            context_vec=context_vec)
        if codes is not None:
            return_dict['codes'] = codes
        return return_dict

    def inner_forward(self, article_sentences, article_sentences_lengths, num_codes, codes, code_description, code_description_length, linearized_codes, linearized_codes_lengths, linearized_descriptions, linearized_descriptions_lengths, *args):
        codes, code_description, code_description_length, linearized_codes, linearized_codes_lengths, linearized_descriptions, linearized_descriptions_lengths = tensor_to_none(codes), tensor_to_none(code_description), tensor_to_none(code_description_length), tensor_to_none(linearized_codes), tensor_to_none(linearized_codes_lengths), tensor_to_none(linearized_descriptions), tensor_to_none(linearized_descriptions_lengths)
        encodings, self_attentions, word_level_attentions = self.clinical_bert_sentences(article_sentences, article_sentences_lengths)
        article_sentences_lengths, num_codes, encodings, self_attentions, word_level_attentions = article_sentences_lengths.to(self.device2), num_codes.to(self.device2), encodings.to(self.device2), self_attentions.to(self.device2), word_level_attentions.to(self.device2)
        b, ns, nl, nh, nt, _ = self_attentions.shape
        traceback_word_level_attentions = ta(self_attentions.mean(3).view(b * ns, nl, nt, nt), attention_vecs=word_level_attentions.view(b * ns, 1, nt)).view(b, ns, nt)
        all_code_embeddings = []
        if codes is not None:
            codes = codes.to(self.device2)
            all_code_embeddings.append(self.code_embeddings(codes))
        if code_description is not None:
            all_code_embeddings.append(self.clinical_bert_sentences(code_description, code_description_length)[0].to(self.device2))
        if linearized_codes is not None:
            all_code_embeddings.append(self.linearized_code_transformer(linearized_codes, linearized_codes_lengths)[0])
        if linearized_descriptions is not None:
            all_code_embeddings.append(self.clinical_bert_sentences(linearized_descriptions, linearized_descriptions_lengths)[0].to(self.device2))
        if self.linear2 is not None:
            code_embeddings = torch.cat(all_code_embeddings, 2)
            code_embeddings = self.linear2(code_embeddings)
        else:
            code_embeddings = all_code_embeddings[0]
        key_padding_mask = (article_sentences_lengths == 0)[:, :encodings.size(1)]
        contextvec, sentence_level_attentions = self.attention(code_embeddings.transpose(0, 1), encodings.transpose(0, 1), encodings.transpose(0, 1), key_padding_mask=key_padding_mask)
        nq, _, emb_dim = contextvec.shape
        word_level_attentions = word_level_attentions.view(b, 1, ns, nt).expand(b, nq, ns, nt)
        traceback_word_level_attentions = traceback_word_level_attentions.view(b, 1, ns, nt).expand(b, nq, ns, nt)
        attention = word_level_attentions * sentence_level_attentions.unsqueeze(3)
        traceback_attention = traceback_word_level_attentions * sentence_level_attentions.unsqueeze(3)
        if self.concatenate_code_embedding:
            encoding = torch.cat([contextvec, code_embeddings.transpose(0, 1)], 2)
            encoding = torch.relu(self.linear3(encoding))
        else:
            encoding = contextvec
        scores = self.linear(encoding).transpose(0, 1)  # shape: (batch, nq, total_num_codes)
        return scores, attention, traceback_attention, contextvec.transpose(0, 1)


class LinearizedCodesTransformer(nn.Module):
    def __init__(self, num_embeddings, d_model=100, num_layers=6, nhead=4):
        super(LinearizedCodesTransformer, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings, d_model)
        self.positional_encodings = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.hidden_size = d_model
        self.num_layers = num_layers
        self.num_heads = nhead

    def forward(self, token_ids, attention_mask):
        b, nl, nh, nt = token_ids.size(0), self.num_layers, self.num_heads, token_ids.size(1)
        outputs = self.transformer_encoder(self.positional_encodings(self.embeddings(token_ids).transpose(0, 1)), src_key_padding_mask=~attention_mask).transpose(0, 1)
        return outputs, outputs[0,0,0]*torch.eye(nt).expand(b, nl, nh, nt, nt)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
