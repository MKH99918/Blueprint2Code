import tiktoken



def token_count(messages, model="gpt-3.5-turbo"):  
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

    tokens_per_message = 4
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens
