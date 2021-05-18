# -*- coding: utf-8 -*-
"""testbert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cRyBMBvxicIJrWeh8IbPTcUOAvxDWKl8
"""

import datasets
import transformers
import rouge_score

# load rouge for validation
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

from transformers import BertTokenizer, EncoderDecoderModel

bert2bert = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16").to("cuda")
tokenizer = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

test_data = datasets.load_dataset("billsum", "3.0.0", split = "test",ignore_verifications=True)#######Change it to local CSVVV

def generate_summary(batch):
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred_summary"] = output_str

    return batch

batch_size = 64  # change to 64 for full evaluation

results = test_data.map(generate_summary, batched=True, batch_size=batch_size)

output = rouge.compute(predictions=results["pred_summary"], references=results["summary"], rouge_types=["rouge2"])["rouge2"].mid
print(output)