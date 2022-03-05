import json
import os
import pandas as pd
import requests
from tqdm import tqdm


class DataFetcher:
    """send requests and get responses from the API.

    Methods
    -------
    request_text(ids_per_request: int = 1000) -> pd.DataFrame
        Request the data from the API, Add it the dataframe, and return it.
    """

    def __init__(self) -> None:
        self.__URL: str = "https://recruitment.aimtechnologies.co/ai-tasks"

    def request_text(self, data_file_path: str, ids_per_request: int = 1000) -> pd.DataFrame:
        """Request the data from the API using ids, Add it the dataframe, and return it.
        
        Parameters
        ----------
        data_file_path: str
            the location of the file containing data ids.
        
        ids_per_request: int
            Number of ids per request, the maximum number allowed is 1000.
            default = 1000

        Returns
        -------
        fetched_data: pd.DataFrame
            A dataframe which contains fetched text and its dialect.
        """

        data_reader = pd.read_csv(data_file_path, 
                                  header = 0, 
                                  dtype = str, 
                                  chunksize = ids_per_request)
        fetched_data = pd.DataFrame(columns = ["text", "dialect"])

        print("Fetching text data:")
        for chunk in tqdm(data_reader):
            ids_list = chunk["id"].to_list()
            ids_json = json.dumps(ids_list)

            response = requests.post(self.__URL, data=ids_json).json()
            chunk["text"] = response.values()
            text_data = chunk[['text','dialect']]

            fetched_data = pd.concat([fetched_data, text_data], ignore_index = True)
        print("Fetching completed Successfully!")

        return fetched_data


if __name__ == "__main__":
    data_fetcher = DataFetcher()
    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets")
    data = data_fetcher.request_text(os.path.join(data_directory, "dialect_id_target.csv"))

    print("Saving The dataframe: ", end="")
    data.to_csv(os.path.join(data_directory, "dialect_dataset.csv"), index=False)
    print("done!")