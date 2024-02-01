import os
import time
import tiktoken
import requests
from datetime import datetime
from datetime import date
from octoai.client import Client
from pinecone import Pinecone

#keys and such
octoai_api_token = os.environ.get("OCTOAI_API_TOKEN")
pc_key = os.environ.get("pc_key")
pc = Pinecone(api_key=pc_key)
client = Client()

def wait_for_input():
    """SOME USER INPUT MECHANISM GOES HERE"""
    if user_initiates_conversation == True:
        conversation_time()

def conversation_time():
    #Conversation setup that adds time/date info to prompt
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    date_string = str (today)
    system_message = {"role": "system", "content": "It is "  + str(current_time) + " on " + str(date_string) + " and you are a friendly robot that enjoys being helpful to your human friend. "}
    user_greeting = {"role": "user", "content": "Hey bot."}
    bot_response = {"role": "assistant", "content": "What's up?"}
    conversation=[]
    conversation.append(system_message)
    conversation.append(user_greeting)
    conversation.append(bot_response)

    #User input section
    user_input = "THIS IS WHAT THE USER ENTERED"
    use_serverless = True
    index = pc.Index("some_pinecone_indexname") #replace with your pinecone index name
    text_field = "text"

    #Embedding the prompt
    payload = {
        "input": user_input,
        "model": "thenlper/gte-large" 
        }
    headers = {
        "Authorization": f"Bearer {octoai_api_token}",
        "Content-Type": "application/json",
        }
    res = requests.post("https://text.octoai.run/v1/embeddings", headers=headers, json=payload)
    xq = res.json()['data'][0]['embedding']
    limit = 500

    #Vector context retrieval
    contexts = []
    res = index.query(vector=xq, top_k=1, include_metadata=True)
    contexts = contexts + [
        x['metadata']['text'] for x in res['matches'] 
        ]
    print (f"Retrieved {len(contexts)} contexts")
    time.sleep(.5)

    #Merge context with prompt and merge into conversation
    prompt_end = (
        f" The following may or may not be relevent information from past conversations. If it is not relevent to this conversation, ignore it:\n\n"
        )
    prompt = (
        user_input +
        "\n\n " +
        prompt_end +
        "\n\n---\n\n".join(contexts[:1])
        )
    conversation.append({"role": "user", "content": prompt})

    #Count tokens from conversation, move context window if max tokens exceeded
    def num_tokens_from_messages(messages, tkmodel="cl100k_base"):
        encoding = tiktoken.get_encoding(tkmodel)
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        conv_history_tokens = num_tokens_from_messages(conversation)
        print("tokens: " + str(conv_history_tokens))
        token_limit = 32000
        max_response_tokens = 300
        while (conv_history_tokens+max_response_tokens >= token_limit):
            del conversation[1] 
            conv_history_tokens = num_tokens_from_messages(conversation)

    #Pass conversation + context + prompt to LLM
    completion = client.chat.completions.create(
        model="mixtral-8x7b-instruct-fp16", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        messages = conversation,
        max_tokens=300,
        presence_penalty=0,
        temperature=0.5,
        top_p=0.9,
        )

    #Append result to conversation
    conversation.append({"role": "assistant", "content": completion.choices[0].message.content})
    response_text = completion.choices[0].message.content + "\n"
    print(response_text)

    if "goodbye" in user_input:
        #Give the user a choice of what the bot will remember
        Bot_Text = "Do you want me to remember this conversation?"
        print(Bot_Text)
        """GET RESPONSE HERE"""
        response = (user_input_2)

        #If no, say goodbye and wait for next conversation
        if "No" in response:
            bot_text = "Ok, I'll talk to you later then. Goodbye."
            print(bot_text)
            wait_for_input()
        
        #otherwise, start storing conversation in memory
        else:
            #Remove system prompt
            del conversation[0]

            #Send conversation to LLM for summarization
            completion = client.chat.completions.create(
            messages = [
                {"role": "system", "content": "Briefly list the key points to remember about the user from this conversation that occurred at "  + str(current_time) + " on " + str(date_string) + ": "}, 
                {"role": "user", "content": str(conversation)},
                ],
            model="mixtral-8x7b-instruct-fp16",
                max_tokens=300,
                temperature=0.3,
                presence_penalty=0,
                top_p=0.9,
                )
            summarization = completion.choices[0].message.content
            
            #Start embedding
            idv = "id" + str(time.time())
            payload = {
                "input": summarization,
                "model": "thenlper/gte-large" 
                }
            headers = {
                "Authorization": f"Bearer {OCTOAI_TOKEN}",
                "Content-Type": "application/json",
                }
            res = requests.post("https://text.octoai.run/v1/embeddings", headers=headers, json=payload)
            
            #Upsert the embedding
            index.upsert(vectors=[{"id":idv, "values":res.json()['data'][0]['embedding'], "metadata":{'text': summarization}}])
            time.sleep(1)
            index.describe_index_stats()

            #return to wait for next conversation
            wait_for_input()
