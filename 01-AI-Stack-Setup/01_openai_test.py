# TEST OPENAI AND SET UP CREDENTIALS
# ***

# Goals:
# 1. Setup OpenAI API Credentials
# 2. Test API connection


# 1.0 OPENAI API ACCESS

#  1. Sign Up or Log In: First, you need to create an account with OpenAI or log into an existing account. Visit OpenAI's API page for this.

#  2. API Key: Once logged in, you will need to access the API key management section to obtain your API key. This key is used to authenticate requests to OpenAI's services.

#  3. Set Rate Limits: Soft and hard limits can be set. Soft will send an email when usage limit has been exceeded. Hard will stop the API from running. 

#  4. Secure Your API Key: We will use a simple YAML file (credentials.yml example is provided).   

# 2.0 TEST API CONNECTION

import yaml
from openai import OpenAI

# OPENAI SETUP:
OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']

client = OpenAI(api_key=OPENAI_API_KEY)

# OpenAI API Models Available
#   Resource: https://platform.openai.com/docs/models/model-endpoint-compatibility
response = client.chat.completions.create(
    model="gpt-3.5-turbo", 
    messages=[
        {"role": "system", "content": "You are a helpful language translating assistant."},
        {"role": "user", "content": "Translate the following English text to French: 'The recent Nike earnings call was upbeat'"},
    ],
    max_tokens=60
)

print(response.choices[0].message.content)

