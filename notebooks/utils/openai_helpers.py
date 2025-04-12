from openai import AzureOpenAI
from dotenv import dotenv_values
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from tqdm import tqdm
import numpy as np

secrets = dotenv_values(".env")
personal_api_key = secrets['AZURE_OPENAI_KEY']
azure_endpoint = secrets['AZURE_OPENAI_ENDPOINT']

client = AzureOpenAI(
  azure_endpoint = azure_endpoint, 
  api_key=personal_api_key,  
  api_version="2024-12-01-preview"
)

output_path = '/home/azureuser/cloudfiles/code/Users/preetams/Dynamic-Evaluation-of-Personaslized-Agents/Results/generated_search_phrases'


logging.basicConfig(filename='openai_logger.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# logging.basicConfig(filename='openai_logger.log', level=logging.ERROR, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# Create a logger object
logger = logging.getLogger(__name__)
# Function to log specific messages
def log_message(message):
    logger.log(logging.INFO, message)

def query_openai_model(prompt, model_name = "gpt4-turbo-0125"):
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        temperature = 0.2,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ],
        response_format={"type": "json_object"}
    )
    usage = response.usage

    return response.choices[0].message.content, usage

def query_openai_model_with_retries(prompt, model_name="gpt4-turbo-0125", max_retries=5):
    retry_count = 0
    wait_time = 1  # Start with 1 second wait time
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}"}
                ],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content, response.usage
        except Exception as e:
            logging.error(f"Error querying prompt '{prompt}': {e}")
            if "rate limit" in str(e).lower():
                retry_count += 1
                logging.info(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2  
            else:
                break  
    return None, None

def query_openai_model_batch(prompts, model_name="gpt4-turbo-0125", max_workers=5):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(query_openai_model_with_retries, prompt, model_name): i for i, prompt in enumerate(prompts)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing prompts"):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                logging.error(f"Error processing a future for prompt index {idx}: {e}")
                results[idx] = (None, None)
    return results

def query_openai_model_batch_save(prompts, model_name="gpt4-turbo-0125", max_workers=5, save_interval=10, save_path=f"{output_path}/results_temp"):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(query_openai_model_with_retries, prompt, model_name): i for i, prompt in enumerate(prompts)}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing prompts")):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                logging.error(f"Error processing a future for prompt index {idx}: {e}")
                log_message(f"Error processing a future for prompt index {idx}: {e}")
                results[idx] = (None, None)
                
            # Save the results every 'save_interval' iterations
            if (i + 1) % save_interval == 0:
                print('Saving intermediate results')
                log_message('Saving intermediate results')
                np.save(f'{save_path}', results)
                
    # Save final results
    np.save(f'{save_path}'+'_final', results)
    
    return results