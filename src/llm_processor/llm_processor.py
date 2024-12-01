import os

import google.generativeai as genai
import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# 環境変数からAPIキーを取得
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google Generative AIの設定
genai.configure(api_key=GOOGLE_API_KEY)


class LLM:
    def __init__(self, provider: str, model: str):
        """
        Args:
            model (str): 使用するモデルの名前
            provider (str): モデルプロバイダ ("gemini" または "openai")
        """
        self.provider = provider
        self.model = model

        if provider == "gemini":
            if not GOOGLE_API_KEY:
                raise ValueError("Google APIキーが設定されていません。")
            self.model = genai.GenerativeModel(model)
        elif provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI APIキーが設定されていません。")
            openai.api_key = OPENAI_API_KEY
        else:
            raise ValueError(
                "無効なプロバイダ名です。'gemini' または 'openai' を指定してください。"
            )

    def create_llm_outputs(self, prompt: pd.DataFrame) -> pd.DataFrame:
        """LLMの出力を生成する

        Args:
            prompt (pd.DataFrame): prompt_creatorで生成されたプロンプト

        Returns:
            pd.DataFrame: {
                target_task: str,
                inject_task: str,
                system_prompt: str,
                inject_task_system_prompt: str,
                injected_data: List[str],
                target_task_data: List[str],
                inject_task_data: List[str],
                target_task_labels: List[str],
                inject_task_labels: List[str],
                injected_data_output: List[str],
                target_task_data_output: List[str],
                inject_task_data_output: List[str]
            }
        """
        prompt["injected_data_output"] = [
            self._generate(f"{prompt["system_prompt"][i]} {prompt["injected_data"][i]}")
            for i in range(len(prompt["injected_data"]))
        ]
        prompt["target_task_data_output"] = [
            self._generate(
                f"{prompt["system_prompt"][i]} {prompt["target_task_data"][i]}"
            )
            for i in range(len(prompt["target_task_data"]))
        ]
        prompt["inject_task_data_output"] = [
            self._generate(
                f"{prompt["inject_task_system_prompt"][i]} {prompt["inject_task_data"][i]}"
            )
            for i in range(len(prompt["inject_task_data"]))
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
                model=self.model,
                prompt=prompt,
                max_tokens=100,  # 必要に応じて調整可能
            )
            return response.choices[0].text.strip()

    def _postprocess(self, prompt: pd.DataFrame) -> pd.DataFrame:
        """LLMの出力を後処理する

        Args:
            prompt (dict): create_llm_outputsで生成されたLLMの出力

        Returns:
            dict: 後処理されたLLMの出力
        """
        # LLM出力に含まれる\nを削除する
        prompt["injected_data_output"] = [
            output.replace("\n", "") for output in prompt["injected_data_output"]
        ]
        prompt["target_task_data_output"] = [
            output.replace("\n", "") for output in prompt["target_task_data_output"]
        ]
        prompt["inject_task_data_output"] = [
            output.replace("\n", "") for output in prompt["inject_task_data_output"]
        ]

        return prompt
