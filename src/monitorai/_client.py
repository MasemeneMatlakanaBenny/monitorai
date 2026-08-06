import requests


class MonitorAI:
    def __init__(self,api_key:str):
        """
        MonitorAI API.
        
        Args:
            api_key (str): the user's generated API-KEY
        """
        self.api_key=api_key
        self.base_url="https://localhost:8000"

        self.headers={
            "Authorization":f"Bearer {api_key}"
        }

    def validate_api(self):
        response=requests.get(
            url=f"{self.base_url}/validate",
            headers=self.headers
        )

        response.raise_for_status()

        return response.json()



