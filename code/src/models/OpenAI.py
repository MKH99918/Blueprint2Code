import os
import dotenv
from openai import OpenAI

from utils.token_count import token_count

dotenv.load_dotenv()


class OpenAIBaseModel(BaseModel):


    def __init__(
        self,
        api_type=None,
        api_base=None,
        api_version=None,
        api_key=None,
        engine_name=None,
        model_name=None,
        temperature=0,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        api_type = api_type or "openai"
        api_key = api_key or "key"
        api_base = api_base or "###"
        model_name = model_name or "gpt-3.5-turbo"

        assert api_key is not None, "API Key must be provided."
        
        if api_type == "azure":
            assert api_base is not None, "API URL must be provided as model config or environment variable (`AZURE_API_BASE`)"
            assert api_version is not None, "API version must be provided as model config or environment variable (`AZURE_API_VERSION`)"


        
        # GPT parameters
        self.model_params = {}
        self.model_params["model"] = model_name
        self.model_params["temperature"] = temperature
        self.model_params["top_p"] = top_p
        self.model_params["max_tokens"] = None
        self.model_params["frequency_penalty"] = frequency_penalty
        self.model_params["presence_penalty"] = presence_penalty


    @staticmethod
    def read_azure_env_vars():
        return {
            "api_version": os.getenv("AZURE_API_VERSION"),
            "api_base": os.getenv("AZURE_API_URL"),
            "api_key": os.getenv("AZURE_API_KEY"),
            "model": os.getenv("AZURE_ENGINE_NAME", os.getenv("ENGINE_NAME")),
        }

    @staticmethod
    def read_openai_env_vars():
        return {
            "api_version": os.getenv("OPENAI_API_VERSION"),
            "api_base": os.getenv("OPENAI_API_BASE"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL"),
        }


class OpenAIModel(OpenAIBaseModel):
    def __init__(
        self,
        api_type=None,
        api_base=None,
        api_version=None,
        api_key=None,
        engine_name=None,
        model_name=None,
        temperature=0.32,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        super().__init__(
            api_type=api_type,
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            engine_name=engine_name,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
    
    def summarize_response(self, response):
        """Returns the first reply from the "assistant", if available"""
        if (
            "choices" in response
            and isinstance(response["choices"], list)
            and len(response["choices"]) > 0
            and "message" in response["choices"][0]
            and "content" in response["choices"][0]["message"]
            and response["choices"][0]["message"]["role"] == "assistant"
        ):
            return response["choices"][0]["message"]["content"]

        return response


    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(self, processed_input: list[dict]):
        """
        OpenAI API ChatCompletion implementation

        Arguments
        ---------
        processed_input : list
            Must be list of dictionaries, where each dictionary has two keys;
            "role" defines a role in the chat (e.g. "system", "user") and
            "content" defines the actual message for that turn

        Returns
        -------
        response : OpenAI API response
            Response from the openai python library

        """
        self.model_params["max_tokens"] = 4096

        response = self.openai.chat.completions.create(
            messages=processed_input,
            **self.model_params
        )

        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens


class GPT4(OpenAIModel):
    def prompt(self, processed_input: list[dict]):
        self.model_params["model"] = "gpt-4o-2024-11-20"
        #self.model_params["model"] = "gpt-4o-mini-2024-07-18"
        return super().prompt(processed_input)


class ChatGPT(OpenAIModel):
    def prompt(self, processed_input: list[dict]):
        self.model_params["model"] = "gpt-3.5-turbo"
        return super().prompt(processed_input)