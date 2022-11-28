from transformers import BertTokenizer, BertModel
from collections import OrderedDict 
import torch
from torch.nn import CosineSimilarity
from sentence_transformers import SentenceTransformer

class BertVectorizer:
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased',
        output_hidden_states = True,)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)



    def bert_text_preparation(self, text):
        """
        Preprocesses text input in a way that BERT can interpret.
        """
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.bert_tokenizer.tokenize(marked_text)
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)

        # convert inputs to tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensor


    def get_bert_embeddings(self, tokens_tensor, segments_tensor):
        """
        Obtains BERT embeddings for tokens, in context of the given sentence.
        """
        # gradient calculation id disabled
        with torch.no_grad():
            # obtain hidden states
            tokens_tensor = tokens_tensor.to(self.device)
            segments_tensor = segments_tensor.to(self.device)
            # print(f"tokens_tensor dev: {tokens_tensor.device} ::: segments tensor dev: {segments_tensor.device}")
            outputs = self.model(tokens_tensor, segments_tensor)
            hidden_states = outputs[2]
        # concatenate the tensors for all layers
        # use "stack" to create new dimension in tensor
        token_embeddings = torch.stack(hidden_states, dim=0)

        # remove dimension 1, the "batches"
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # swap dimensions 0 and 1 so we can loop over tokens
        token_embeddings = token_embeddings.permute(1,0,2)

        # intialized list to store embeddings
        token_vecs_sum = []

        # "token_embeddings" is a [Y x 12 x 768] tensor
        # where Y is the number of tokens in the sentence

        # loop over tokens in sentence
        for token in token_embeddings:

            # "token" is a [12 x 768] tensor

            # sum the vectors from the last four layers
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)

        return token_vecs_sum


    def get_context_embeddings(self, block):

        context_embeddings = []
        context_tokens = []
        tokenized_text, tokens_tensor, segments_tensors = self.bert_text_preparation(block)
        list_token_embeddings = self.get_bert_embeddings(tokens_tensor, segments_tensors)

        # make ordered dictionary to keep track of the position of each word
        tokens = OrderedDict()

        # loop over tokens in sensitive sentence
        for token in tokenized_text[1:-1]:
            # keep track of position of word and whether it occurs multiple times
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1

            # compute the position of the current token
            token_indices = [i for i, t in enumerate(tokenized_text) if t == token]
            current_index = token_indices[tokens[token]-1]

            # get the corresponding embedding
            token_vec = list_token_embeddings[current_index]
            
            # save values
            context_tokens.append(token)
            context_embeddings.append(token_vec.cpu())
        
        return context_embeddings


    def block_similarity(self, b1, b2):
        b1_embedding = self.get_context_embeddings(b1)
        b2_embedding = self.get_context_embeddings(b2)

        b1_context_tensor = torch.stack(b1_embedding)
        b2_context_tensor = torch.stack(b2_embedding)
        cos = CosineSimilarity(dim=1, eps=1e-6)
        block_sim = torch.linalg.norm(cos(b1_context_tensor, b2_context_tensor))
        return block_sim


class SbertVectorizer:
    def __init__(self):
        '''
        The all-mpnet-base-v2 model provides the best quality, 
        while all-MiniLM-L6-v2 is 5 times faster and still offers good quality 
        '''
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def block_similarity(self, b1, b2):
        b1_embedding = self.model.encode(b1)
        b2_embedding = self.model.encode(b2)
        cos = CosineSimilarity(dim=0, eps=1e-6)
        b1_embedding_tensor = torch.from_numpy(b1_embedding)
        b2_embedding_tensor = torch.from_numpy(b2_embedding)
        return cos(b1_embedding_tensor[0], b2_embedding_tensor[1])

class PhraseBertVectorizer:
    def __init__(self) -> None:
        self.model = SentenceTransformer('phrase-bert-model/pooled_context_para_triples_p=0.8/0_BERT')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def block_similarity(self, b1, b2):
        b1_embedding = self.model.encode(b1)
        b2_embedding = self.model.encode(b2)
        cos = CosineSimilarity(dim=0, eps=1e-6)
        b1_embedding_tensor = torch.from_numpy(b1_embedding)
        b2_embedding_tensor = torch.from_numpy(b2_embedding)
        return cos(b1_embedding_tensor[0], b2_embedding_tensor[1])





