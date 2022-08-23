import os
import pickle
import argparse
from data.tatqa_dataset import TagTaTQATestReader, TagTaTQAReader
#from transformers.tokenization_roberta import RobertaTokenizer
from transformers import RobertaTokenizer,T5Tokenizer,ElectraTokenizer
from transformers import BertTokenizer,AutoTokenizer,DebertaV2Tokenizer
from transformers import XLMRobertaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--passage_length_limit", type=int, default=463)
parser.add_argument("--question_length_limit", type=int, default=46)
parser.add_argument("--encoder", type=str, default="bert")
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--roberta_model", type=str, default='')

args = parser.parse_args()

if args.encoder == 'roberta':
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)
    sep = '<s>'
elif args.encoder == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    sep = '[SEP]'
elif args.encoder == 'finbert':
    #tokenizer = BertTokenizer.from_pretrained(args.input_path + "/finbert")
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    sep = '[SEP]'
elif args.encoder == "deberta-v3-large":
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
    sep = '[SEP]'
elif args.encoder == "deberta-v2-xlarge":
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
    sep = '[SEP]'
elif args.encoder == "t5-11b":
    tokenizer = T5Tokenizer.from_pretrained("t5-11b")
    sep = '[SEP]'
elif args.encoder == "t5-3b":
    tokenizer = T5Tokenizer.from_pretrained("t5-3b")
    sep = '[SEP]'
elif args.encoder == "albert-xxlarge-v2":
    tokenizer = T5Tokenizer.from_pretrained("albert-xxlarge-v2")
    sep = '[SEP]'
elif args.encoder == "xlm-roberta-large":
    tokenizer = XLMRobertaTokenizer.from_pretrained("/data/pjli/workspace/gitcodes/L2I/xlm-roberta-large/")
    sep = '[SEP]'
elif args.encoder == "electra-large-discriminator":
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-large-discriminator")
    sep = '[SEP]'








if args.mode == 'dev':
    data_reader = TagTaTQATestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    data_mode = ["dev"]
else:
    data_reader = TagTaTQAReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    data_mode = ["train"]

data_format = "tatqa_and_hqa_field_{}.json"
print(f'==== NOTE ====: encoder:{args.encoder}, mode:{args.mode}')

for dm in data_mode:
    dpath = os.path.join(args.input_path, data_format.format(dm))
    data = data_reader._read(dpath)
    print("skipping number: ", data_reader.skip_count)
    data_reader.skip_count = 0
    print("Save data to {}.".format(os.path.join(args.output_dir, f"tagop_{args.encoder}_cached_{dm}.pkl")))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, f"tagop_{args.encoder}_cached_{dm}.pkl"), "wb") as f:
        pickle.dump(data, f)
