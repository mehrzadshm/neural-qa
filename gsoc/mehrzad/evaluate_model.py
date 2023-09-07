from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric
import pandas as pd
import torch

import argparse
import logging
import csv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--data_path", type=str, default="/")
    parser.add_argument("--metric_name", type=str, default="bleu")
    parser.add_argument("--input_column_name", type=str, default="prompt")
    parser.add_argument("--output_column_name", type=str, default="completion")
    # parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--report_file_name", type=str, default="eval_results")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

 
    
def read_dataset(path):
    return pd.read_csv(path)



def generate_SPARQL(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=250,
        num_beams=5,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )
    
    decoded_output = tokenizer.decode(outputs[0])
    
    return decoded_output




def calculate_bleu(actual, generated, metric_name):
    bleu_metric = load_metric(metric_name)
    bleu_score = bleu_metric.compute(predictions=[generated.split()], references=[[actual.split()]])
    return bleu_score['bleu']




class SPARQLPostProcessor:
    
    ## TODO

    # deal with spaces


    # for now, remove things based on "Answer:" token
    def remove_from_token(self, generated_output, reference_token="Answer:"):
        return generated_output.replace(reference_token, '').strip()


    # def remove_prompt(self, generated_output, reference_token="Answer:"):
    #     return generated_output.replace(reference_token, '').strip()


    
    def post_process(self, query):
        query = self.remove_from_token(query)
        #
        #
        return query



def main():
    
    args = get_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)


    data = read_dataset(args.data_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    

    logging.info(f"Loading tokenizer and model on {device}.\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token   # Set pad token to EOS token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, use_auth_token=True).to(device)



    print(f'\nCalculating scores with <{args.metric_name}> as evaluation metric\n...'+"\n"*2)
    
    
    total_bleu_score = 0

    post_processor = SPARQLPostProcessor()

    with open(f'{args.report_file_name}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row
        csvwriter.writerow(['Prompt', 'Actual SPARQL', 'Generated SPARQL', 'BLEU Score'])
        

        for idx, row in data.iterrows():
            prompt, actual_sparql = row[args.input_column_name], row[args.output_column_name]
            generated_sparql = generate_SPARQL(model, tokenizer, prompt)
            generated_sparql = generated_sparql[len(prompt):]
            # generated_sparql = post_processor.post_process(generated_sparql[len(prompt):])

            bleu_score = calculate_bleu(actual_sparql, generated_sparql, args.metric_name)

            csvwriter.writerow([prompt, actual_sparql, generated_sparql, bleu_score])

            total_bleu_score += bleu_score

    average_bleu_score = total_bleu_score / len(data)

    print("\n"+"*"*80 +f"\nResults written to eval_results.csv")
    print(f"\nAverage BLEU score for {len(data)} test samples: ", average_bleu_score)


    # delete model & tokenizer and clear cache 
    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        

if __name__ == "__main__" :
    main()            

