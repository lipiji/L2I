import json
import random

with open("tatqa_and_hqa_field_all.json",'r') as load_f:
  jsonob = json.load(load_f)

random.shuffle(jsonob)
train = jsonob[:2000]
dev = jsonob[2000:]

with open("tatqa_and_hqa_field_train.json","w") as dump_f:
  json.dump(train, dump_f, ensure_ascii=False)

with open("tatqa_and_hqa_field_dev.json","w") as dump_f:
  json.dump(dev, dump_f, ensure_ascii=False)


