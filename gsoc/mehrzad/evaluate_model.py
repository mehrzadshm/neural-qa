from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric
import pandas as pd
import torch

import re
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
        max_length=150,
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
    # Remove unwanted tokens
    def remove_tokens(self, generated_output, tokens_to_remove=["Answer:", "<|endoftext|>"]):  # Replace with the actual EOS token
        for token in tokens_to_remove:
            generated_output = generated_output.replace(token, '').strip()
        return generated_output


    # custom spacing rules to algin with nspm dataset
    def apply_spacing_rules(self, query):
    
        # Remove redundant spaces
        query = re.sub(r' +', ' ', query)
        query = query.strip()
        
        # Ensure space before "?"
        query = re.sub(r'(?<=[^\s])\?', ' ?', query)
        
        # Ensure no space after "?"
        query = re.sub(r'\?[\s]+', '?', query)
        
        # Ensure space before "}"
        query = re.sub(r'(?<=[^\s])}', ' }', query)
        
        # Ensure space before "."
        query = re.sub(r'(?<=[^\s])\.', ' .', query)
        
        # Ensure space after "."
        query = re.sub(r'\.(?=[^\s])', '. ', query)
        
        # Ensure no space before and after "{"
        query = re.sub(r'\s?{\s?', '{', query)
        
        # Ensure no space before and after ":"
        query = re.sub(r'\s?:\s?', ':', query)
        
        return query
    

    def post_process(self, query):
        query = self.remove_tokens(query)
        query = self.apply_spacing_rules(query)
        
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
    
    
    total_bleu_score_raw, total_bleu_score_postproccesed = 0, 0

    post_processor = SPARQLPostProcessor()
    

    with open(f'{args.report_file_name}.csv', 'w', newline='') as csvfile:
        
        csvwriter = csv.writer(csvfile)
        # Write the header row
        csvwriter.writerow(['Prompt', 'Actual SPARQL', 'Generated SPARQL', 'BLEU Score (Generated)', 'Post-Processed SPARQL', 'BLEU Score (Post-Processed)'])
    
        # Initialize total BLEU scores
        total_bleu_score_raw = 0
        total_bleu_score_postprocessed = 0
        
        for idx, row in data.iterrows():
            try:
                prompt, actual_sparql = row[args.input_column_name], row[args.output_column_name]
                
                # Generate SPARQL
                generated_sparql = generate_SPARQL(model, tokenizer, prompt)
                generated_sparql = generated_sparql[len(prompt):]  # Removing the prompt from the generated SPARQL

                # Calculate BLEU Score for generated SPARQL
                bleu_score_generated = calculate_bleu(actual_sparql, generated_sparql, args.metric_name)
                total_bleu_score_raw += bleu_score_generated

                # Post-process the SPARQL
                post_processed_sparql = post_processor.post_process(generated_sparql)

                # Calculate BLEU Score for post-processed SPARQL
                bleu_score_post_processed = calculate_bleu(actual_sparql, post_processed_sparql, args.metric_name)
                total_bleu_score_postprocessed += bleu_score_post_processed

                # Write Results to CSV
                csvwriter.writerow([prompt, actual_sparql, generated_sparql, bleu_score_generated, post_processed_sparql, bleu_score_post_processed])

                # Log Progress
                if args.verbose and idx % 100 == 0:  # adjust based on how frequently you want to log
                    logging.info(f"Processed {idx} rows. BLEU Score - Generated: {bleu_score_generated}, Post-Processed: {bleu_score_post_processed}")  


            except Exception as e:
                print(f"Error processing row {idx}: {e}")
 

    #average_bleu_score_raw = total_bleu_score_raw / len(data)
    #average_bleu_score_postproccesed = total_bleu_score_postproccesed / len(data)

    
    print("\n" + "*"*80 + f"\nResults written to {args.report_file_name}.csv")

    #print(f"\nAverage BLEU score for raw generated queries with {len(data)} test samples: ", average_bleu_score_raw)
    #print(f"\nAverage BLEU score with post-proccesing: ", average_bleu_score_postproccesed)


    # delete model & tokenizer and clear cache 
    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        

if __name__ == "__main__" :
    main()            

