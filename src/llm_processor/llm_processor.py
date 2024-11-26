import os
from dotenv import load_dotenv
import google.generativeai as genai
import openai

load_dotenv()

# 環境変数からAPIキーを取得
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google Generative AIの設定
genai.configure(api_key=GOOGLE_API_KEY)

class LLM:
    def __init__(self,provider: str, model_name: str):
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
            raise ValueError("無効なプロバイダ名です。'gemini' または 'openai' を指定してください。")

    def generate(self, prompt: str) -> str:
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
                max_tokens=100  # 必要に応じて調整可能
            )
            return response.choices[0].text.strip()
