from tqdm.auto import tqdm
import torch
from typing import List
from spatialdata import SpatialData


class EmbeddingSimilarityRetriever:
    def __init__(self, spatial_data_list: List[SpatialData], dataset_names: List[str]):
        """
        Initializes the retriever with a list of SpatialData objects and their corresponding dataset names.

        Args:
            spatial_data_list: List of SpatialData objects.
            dataset_names: List of dataset names corresponding to the SpatialData objects.
        """
        self.spatial_data_list = spatial_data_list
        self.dataset_names = dataset_names

    @staticmethod
    def compute_scores(test_embedding: torch.Tensor, candidate_embeddings: torch.Tensor) -> List[float]:
        """
        Computes cosine similarity between a test embedding and candidate embeddings.

        Args:
            test_embedding: The embedding of the test sample as a tensor.
            candidate_embeddings: A tensor of candidate embeddings.

        Returns:
            A list of similarity scores.
        """
        scores = torch.nn.functional.cosine_similarity(test_embedding, candidate_embeddings, dim=1)
        return scores.numpy().tolist()

    def retrieve_similarity_scores(self, test_embedding: torch.Tensor) -> dict:
        """
        Iterates through the SpatialData objects, computes similarity scores, and returns a mapping of unique identifiers to scores.

        Args:
            test_embedding: The test embedding as a tensor.

        Returns:
            A dictionary with unique identifiers as keys and similarity scores as values.
        """
        similarity_mapping = {}

        for spatial_data, dataset_name in zip(self.spatial_data_list, self.dataset_names):
            embeddings = torch.tensor(spatial_data.obsm['embeddings']) 
            # or embeddings = torch.stack(spatial_data.obsm['embeddings']).squeeze(1)
            sim_scores = self.compute_scores(test_embedding, embeddings)

            candidate_ids = [f"{dataset_name}_{i}" for i in range(len(embeddings))]
            similarity_mapping.update(dict(zip(candidate_ids, sim_scores)))

        return similarity_mapping

    def fetch_similar(self, test_embedding: torch.Tensor, top_k: int = 5) -> dict:
        """
        Fetches the top_k similar embeddings from the SpatialData objects.

        Args:
            test_embedding: The embedding of the test sample as a tensor.
            top_k: Number of top results to return.

        Returns:
            A dictionary containing the top_k similar patches with metadata.
        """
        all_scores = self.retrieve_similarity_scores(test_embedding)

        # Sort the mapping dictionary and select top_k entries.
        similarity_mapping_sorted = dict(
            sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        )

        id_entries = list(similarity_mapping_sorted.keys())[:top_k]
        selected_candidate_datasets = [entry.split("_")[0] for entry in id_entries]
        patches_idx = [int(entry.split("_")[-1]) for entry in id_entries]
        similarity_scores = list(similarity_mapping_sorted.values())[:top_k]

        results_dict = {}
        for i, entry in enumerate(id_entries):
            results_dict[entry] = {
                'dataset_name': selected_candidate_datasets[i],
                'patch_id': patches_idx[i],
                'similarity_score': similarity_scores[i]
            }

        return results_dict


if __name__ == "__main__":
    test_set_embedding = torch.tensor(test_sample.obsm['embeddings'][10])  
    spatial_data_list = [spatial_data_obj1, spatial_data_obj2]
    dataset_names = ["Dataset1", "Dataset2"]

    retriever = EmbeddingSimilarityRetriever(spatial_data_list, dataset_names)
    results_dict = retriever.fetch_similar(test_set_embedding, top_k=5)
