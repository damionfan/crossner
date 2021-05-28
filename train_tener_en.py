from models.TENER import TENER
from fastNLP.embeddings import CNNCharEmbedding
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from fastNLP import SpanFPreRecMetric, BucketSampler
from fastNLP.io.pipe.conll import OntoNotesNERPipe
from fastNLP.embeddings import StaticEmbedding, LSTMCharEmbedding, BertEmbedding
from modules.TransformerEmbedding import TransformerCharEmbed, TransformerCharEmbed1
from modules.pipe import Conll2003NERPipe

import argparse
from modules.callbacks import EvaluateCallback


from typing import List

import torch
from torch import nn as nn

from fastNLP.embeddings.embedding import TokenEmbedding


class Cross_Attention(nn.Module):
    """
        Cross Attention
    """
    def __init__(self, dim1,dim2,hidden_dim, both=True):
        super(Cross_Attention,self).__init__()
        # cross attention
        self.f1_bigru = nn.LSTM(input_size=dim1, hidden_size=hidden_dim//2, num_layers=1,bidirectional=True, batch_first=True, bias=True)
        self.f2_bigru = nn.LSTM(input_size=dim2, hidden_size=hidden_dim//2, num_layers=1,bidirectional=True, batch_first=True, bias=True)

        self.both = both

    def forward(self,f1,f2):
        f1,_ = self.f1_bigru(f1)
        #f1 = torch.tanh(f1)
        f2,_ = self.f2_bigru(f2)
        #f2 = torch.tanh(f2)

        m1 = torch.bmm(f1,f2.permute(0,2,1))
        m2 = torch.bmm(f2,f1.permute(0,2,1))

        n1 = torch.nn.functional.softmax(m1,dim=-1)
        n2 = torch.nn.functional.softmax(m2,dim=-1)

        o1 = torch.bmm(n1,f2)
        o2 = torch.bmm(n2,f1)

        a1 = torch.mul(o1,f1)
        a2 = torch.mul(o2,f2)

        if self.both:
            return torch.cat((a1,a2),dim=-1)
        return a2 # 返回word字符


class StackEmbedding(TokenEmbedding):
    r"""
    支持将多个embedding集合成一个embedding。
    Example::
        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import StaticEmbedding, StackEmbedding
        >>> vocab =  Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed_1 = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50d', requires_grad=True)
        >>> embed_2 = StaticEmbedding(vocab, model_dir_or_name='en-word2vec-300', requires_grad=True)
        >>> embed = StackEmbedding([embed_1, embed_2])
    """
    
    def __init__(self, embeds: List[TokenEmbedding], word_dropout=0, dropout=0):
        r"""
        
        :param embeds: 一个由若干个TokenEmbedding组成的list，要求每一个TokenEmbedding的词表都保持一致
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。不同embedidng会在相同的位置
            被设置为unknown。如果这里设置了dropout，则组成的embedding就不要再设置dropout了。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        """
        vocabs = []
        for embed in embeds:
            if hasattr(embed, 'get_word_vocab'):
                vocabs.append(embed.get_word_vocab())
        _vocab = vocabs[0]
        for vocab in vocabs[1:]:
            #if _vocab!=vocab:
            assert vocab == _vocab, "All embeddings in StackEmbedding should use the same word vocabulary."

        super(StackEmbedding, self).__init__(_vocab, word_dropout=word_dropout, dropout=dropout)
        assert isinstance(embeds, list)
        for embed in embeds:
            assert isinstance(embed, TokenEmbedding), "Only TokenEmbedding type is supported."
        self.embeds = nn.ModuleList(embeds)
        self._embed_size = sum([embed.embed_size for embed in self.embeds])
        
        self.ca1 = Cross_Attention(embeds[0].embed_size,embeds[1].embed_size,embeds[1].embed_size) # cnn
        self.ca2 = Cross_Attention(embeds[0].embed_size,embeds[2].embed_size,embeds[2].embed_size) # local attention
        self.ca3 = Cross_Attention(embeds[0].embed_size,embeds[3].embed_size,embeds[3].embed_size) # global attention
        
        self.liner = nn.Linear(embeds[1].embed_size+embeds[2].embed_size+embeds[3].embed_size,embeds[0].embed_size)

    def append(self, embed: TokenEmbedding):
        r"""
        添加一个embedding到结尾。
        :param embed:
        :return:
        """
        assert isinstance(embed, TokenEmbedding)
        _check_vocab_has_same_index(self.get_word_vocab(), embed.get_word_vocab())
        self._embed_size += embed.embed_size
        self.embeds.append(embed)
        return self
    
    def pop(self):
        r"""
        弹出最后一个embed
        :return:
        """
        embed = self.embeds.pop()
        self._embed_size -= embed.embed_size
        return embed
    
    @property
    def embed_size(self):
        r"""
        该Embedding输出的vector的最后一维的维度。
        :return:
        """
        return self._embed_size
    
    def forward(self, words):
        r"""
        得到多个embedding的结果，并把结果按照顺序concat起来。
        :param words: batch_size x max_len
        :return: 返回的shape和当前这个stack embedding中embedding的组成有关
        """
        outputs = []
        words = self.drop_word(words)
        for embed in self.embeds:
            x = embed(words)
            # print('###############################\n',x.size())
            outputs.append(x)
        '''
        0 word 
        1 cnn
        2 local
        3 gloabl 
        '''
        cnn = self.ca1(outputs[0],outputs[1])
        local = self.ca2(outputs[0],outputs[2])
        gloabl = self.ca3(outputs[0],outputs[3])

        cnn_w = cnn[:,:,:-self.embeds[1].embed_size]
        local_w = local[:,:,:-self.embeds[2].embed_size]
        global_w = gloabl[:,:,:-self.embeds[3].embed_size]

        word_char = torch.cat((cnn_w,local_w,global_w),-1)

        char_infor = torch.cat((cnn[:,:,self.embeds[1].embed_size:],local[:,:,self.embeds[2].embed_size:],gloabl[:,:,self.embeds[3].embed_size:]),-1)

        word_char = self.liner(word_char)
        outputs = torch.cat((word_char,char_infor),dim=-1)
        #output1 = self.ca(outputs[0],outputs[1])
        #output2 = self.cas(outputs[0],outputs[2])

        #output1 = self.linear1(output1)
        #output2 = self.linear2(output2)

        #outputs = self.dropout(torch.cat((outputs[0],output1,output2), dim=-1))
        #outputs = self.dropout(torch.cat(outputs,dim=-1))
        return outputs



device = 0
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='en-ontonotes', choices=['conll2003', 'en-ontonotes'])

args = parser.parse_args()

dataset = args.dataset

if dataset == 'conll2003':
    n_heads = 14
    head_dims = 128
    num_layers = 2
    lr = 0.0009#0.0009 # 3e-4
    attn_type = 'adatrans'
    char_type = 'cnn' #'cnn'
elif dataset == 'en-ontonotes':
    n_heads =  8
    head_dims = 96
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    char_type = 'adatrans'

pos_embed = None

#########hyper
batch_size = 32#16
warmup_steps = 0.01
after_norm = 1
model_type = 'transformer'
normalize_embed = True
#########hyper

dropout=0.5
fc_dropout=0.4

encoding_type = 'bioes'
name = 'caches/{}_{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, char_type, normalize_embed)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)



@cache_results(name, _refresh=False)
def load_data():
    # 替换路径
    if dataset == 'conll2003':
        # conll2003的lr不能超过0.002
        paths = {'test': "../data/NCBI-disease-IOB/test.tsv",
                 'train': "../data/NCBI-disease-IOB/train.tsv",
                 'dev': "../data/NCBI-disease-IOB/test.tsv"}
        data = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(paths)
    elif dataset == 'en-ontonotes':
        # 会使用这个文件夹下的train.txt, test.txt, dev.txt等文件
        paths = '../data/en-ontonotes/english'
        data = OntoNotesNERPipe(encoding_type=encoding_type).process_from_file(paths)
    char_embed = None
    if char_type == 'cnn':
        char_embed1 = CNNCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, filter_nums=[30],
                                      kernel_sizes=[3], word_dropout=0, dropout=0.3, pool_method='max'
                                      , include_word_start_end=False, min_char_freq=1)
        
        char_embed2 = TransformerCharEmbed(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                 dropout=0.3, pool_method='max', activation='relu',
                 min_char_freq=2, requires_grad=True, include_word_start_end=False,
                 char_attn_type="adatrans", char_n_head=3, char_dim_ffn=60, char_scale='adatrans'=='naive',
                 char_dropout=0.15, char_after_norm=True)
        
        char_embed3 = TransformerCharEmbed1(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                 dropout=0.3, pool_method='max', activation='relu',
                 min_char_freq=2, requires_grad=True, include_word_start_end=False,
                 char_attn_type="adatrans", char_n_head=3, char_dim_ffn=60, char_scale='adatrans'=='naive',
                 char_dropout=0.15, char_after_norm=True)
        
    elif char_type in ['adatrans', 'naive']:
        char_embed = TransformerCharEmbed(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                 dropout=0.3, pool_method='max', activation='relu',
                 min_char_freq=2, requires_grad=True, include_word_start_end=False,
                 char_attn_type=char_type, char_n_head=3, char_dim_ffn=60, char_scale=char_type=='naive',
                 char_dropout=0.15, char_after_norm=True)
    elif char_type == 'lstm':
        char_embed = LSTMCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                 dropout=0.3, hidden_size=100, pool_method='max', activation='relu',
                 min_char_freq=2, bidirectional=True, requires_grad=True, include_word_start_end=False)
    #word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
     #                            model_dir_or_name='en-glove-6b-100d',
     #                            requires_grad=True, lower=True, word_dropout=0, dropout=0.5,
      #                           only_norm_found_vector=normalize_embed)
    word_embed = BertEmbedding(vocab=data.get_vocab('words'), model_dir_or_name='./bert/', layers='-2,-1', pool_method='max')

    if char_embed is not None:
        embed = StackEmbedding([word_embed, char_embed1,char_embed2,char_embed3], dropout=0, word_dropout=0.02)
    else:
        word_embed.word_drop = 0.02
        embed = word_embed

    data.rename_field('words', 'chars')
    return data, embed

data_bundle, embed = load_data()
print(data_bundle)

model = TENER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
                       d_model=d_model, n_head=n_heads,
                       feedforward_dim=dim_feedforward, dropout=dropout,
                        after_norm=after_norm, attn_type=attn_type,
                       bi_embed=None,
                        fc_dropout=fc_dropout,
                       pos_embed=pos_embed,
              scale=attn_type=='transformer')

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=lr)
callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))

if warmup_steps>0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])
trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer, batch_size=batch_size, sampler=BucketSampler(),
                  num_workers=2, n_epochs=50, dev_data=data_bundle.get_dataset('dev'),
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type,only_gross=False),
                  dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=100, save_path=None)
trainer.train(load_best_model=False)
from fastNLP import Tester

tester = Tester(data_bundle.get_dataset('test'), model, metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target')))
tester.test()
