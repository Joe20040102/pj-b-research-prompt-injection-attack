import os

import google.generativeai as genai
import openai
from dotenv import load_dotenv

load_dotenv()

# 環境変数からAPIキーを取得
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google Generative AIの設定
genai.configure(api_key=GOOGLE_API_KEY)


class LLM:
    def __init__(self, provider: str, model_name: str):
        """
        Args:
            model_name (str): 使用するモデルの名前
            provider (str): モデルプロバイダ ("gemini" または "openai")
        """
        self.provider = provider
        self.model_name = model_name

        if provider == "gemini":
            if not GOOGLE_API_KEY:
                raise ValueError("Google APIキーが設定されていません。")
            self.model = genai.GenerativeModel(model_name)
        elif provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI APIキーが設定されていません。")
            openai.api_key = OPENAI_API_KEY
        else:
            raise ValueError(
                "無効なプロバイダ名です。'gemini' または 'openai' を指定してください。"
            )

    def create_llm_outputs(self, prompt: dict) -> dict:
        """LLMの出力を生成する

        Args:
            prompt (dict): prompt_creatorで生成されたプロンプト

        Returns:
            dict: {
                target_task: str,
                injected_task: str,
                system_prompt: str,
                injected_system_prompt: str,
                injected_data: List[str],
                target_task_data: List[str],
                injected_task_data: List[str],
                target_task_labels: List[str],
                injected_task_labels: List[str],
                injected_data_output: List[str],
                target_task_data_output: List[str],
                injected_task_data_output: List[str]
            }
        """
        prompt["injected_data_output"] = [
            self._generate(f"{prompt["system_prompt"]} {prompt["injected_data"][i]}")
            for i in range(len(prompt["injected_data"]))
        ]
        prompt["target_task_data_output"] = [
            self._generate(f"{prompt["system_prompt"]} {prompt["target_task_data"][i]}")
            for i in range(len(prompt["target_task_data"]))
        ]
        prompt["injected_task_data_output"] = [
            self._generate(
                f"{prompt["injected_system_prompt"]} {prompt["injected_task_data"][i]}"
            )
            for i in range(len(prompt["injected_task_data"]))
        ]
        self._postprocess(prompt)
        return prompt

    def _generate(self, prompt: str) -> str:
        """プロンプトに基づいてタスクを処理する
        Args:
            prompt (str): 入力プロンプト
        Returns:
            str: モデルが生成したテキスト
        """
        if self.provider == "gemini":
            response = self.model.generate_content(prompt)
            return response.text
        elif self.provider == "openai":
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=100,  # 必要に応じて調整可能
            )
            return response.choices[0].text.strip()
