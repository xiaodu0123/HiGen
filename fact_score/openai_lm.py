import openai
import time
import os
from dotenv import load_dotenv


class OpenAIModel(object):
    def __init__(self, model_name):
        load_dotenv(".env")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        self.model_name = model_name

    def generate(self, prompt, max_output_tokens):
        client = openai.OpenAI(api_key=self.api_key)
        wait_time = 1
        while True:
            try:
                response = client.chat.completions.create(
                    messages=[{'role': 'user', 'content': prompt}],
                    model=self.model_name,
                    max_tokens=max_output_tokens,
                    temperature=0.7,
                    top_p=0.9,
                )
                break
            except openai.RateLimitError as e:
                print(f'{e}: waiting {wait_time}s')
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, wait_time + 30)
        return response.choices[0].message.content
