from colpali_engine.interpretability.similarity_maps import plot_similarity_map
from colpali_engine.interpretability.similarity_map_utils import get_similarity_maps_from_embeddings
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device
from pdf2image import convert_from_path
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch

DOCUMENT_EMBEDDING_FILENAME = "page_embeddings.pt"
DOCUMENT_MASK_FILENAME = "page_masks.pt"

def generate_document_and_query(document_path, query, progress_callback=None):
    # This function syncs the UI with the state document_path and query, and also
    # performs the necessary calculations or loads the previously calculated data with respect
    # to the document_path and query in the state
    if not document_path or document_path == "<Not Selected>":
        return (None, None)

    print(f"Loading document at path: {document_path}")
    document_name = (document_path.split("/")[-1]).split(".")[0]
    # make directory based on document name
    try:
        print("Document data directory does not exist, generating directory and document image files...")
        os.mkdir(document_name)
        convert_from_path(document_path, output_folder=document_name, output_file=document_name, fmt="png")
    except FileExistsError: 
        print("Document data directory already exists, attempting to load existing data.")

    base_page_paths = []
    encoding_file = None
    for file in os.listdir(document_name):
        if file == DOCUMENT_EMBEDDING_FILENAME:
            encoding_file = file
            continue
        if file.startswith(document_name) & file.endswith(".png"):
            image_path = os.path.join(document_name, file)
            base_page_paths.append(image_path)

    base_page_paths.sort()
    if len(base_page_paths) == 0:
        return (None, None)

    if (not query) | (query == "<None>"):
        return (base_page_paths, None)
    else:
        print(f"Query exists. Embedding or loading existing checkpoint for document and query")

        # Load/generate document encodings
        page_images = []
        print(f"Opening {len(base_page_paths)} images...")
        for path in base_page_paths:
            try:
                page_images.append(Image.open(path))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        
        if not encoding_file:
            print("Previous embedding file not found, generating new embeddings.")
            (model, processor, device) = get_model_and_processor()

            # Process first image to determine sizes
            processed_first_image = processor.process_images([page_images[0]]).to(device)
            with torch.no_grad():
                embedded_first_image = model(**processed_first_image)

            image_shapes = embedded_first_image.shape
            image_embeddings = torch.zeros([len(page_images), image_shapes[1], image_shapes[2]], dtype=torch.bfloat16).to(device)
            mask_shapes = processor.get_image_mask(processed_first_image).shape
            image_masks = torch.zeros([len(page_images), mask_shapes[1]], dtype=torch.bool).to(device)

            # Vectorizing the images and recording them in tensor...
            for i, image in enumerate(page_images):
                print(f"Embedding image {i}")
                processed_image = processor.process_images([image]).to(device)
                mask = processor.get_image_mask(processed_first_image)
                image_masks[i, :] = mask
                with torch.no_grad():
                    embedded_image = model(**processed_image)
                    image_embeddings[i, :, :] = embedded_image
            torch.save(image_embeddings, os.path.join(document_name, DOCUMENT_EMBEDDING_FILENAME))
            torch.save(image_masks, os.path.join(document_name, DOCUMENT_MASK_FILENAME))
        else:
            print("Previous embedding file found, loading embeddings from file")
            image_embeddings = torch.load(os.path.join(document_name, DOCUMENT_EMBEDDING_FILENAME))
            image_masks = torch.load(os.path.join(document_name, DOCUMENT_MASK_FILENAME))

        
        print(f"Embedding query: {query}")
        with torch.no_grad():
            (model, processor, device) = get_model_and_processor()
            query_embeddings = model(**processor.process_queries([query]).to(device))

        relevancy_scores = processor.score_multi_vector(query_embeddings, image_embeddings)[0].numpy()

        n_patches = processor.get_n_patches(image_size=page_images[0].size, spatial_merge_size=model.spatial_merge_size)
        
        print(query_embeddings.repeat([len(page_images), 1, 1]).shape)
        similarity_maps = get_similarity_maps_from_embeddings(image_embeddings, query_embeddings.repeat([len(page_images), 1, 1]), n_patches, image_masks)
        summed_maps = []
        for map in similarity_maps:
            summed_map = map.sum(0)
            summed_maps.append(summed_map)
        print(len(summed_maps))
        print(summed_maps[0].shape)

        min_relevance = min(relevancy_scores)
        max_relevance = max(relevancy_scores)

        highlighted_page_paths = []
        print(f"Generating {len(page_images)} highlighted pages...")
        plt.ioff()
        for i, image in enumerate(page_images):
            print(f"hihi {((relevancy_scores[i] - min_relevance)/(max_relevance - min_relevance) + 0.1)}")
            tensor_max = torch.max(summed_maps[i])
            summed_maps[i][0][0] =  tensor_max / ((relevancy_scores[i] - min_relevance)/(max_relevance - min_relevance) + 0.1)
            (fig, ax) = plot_similarity_map(image, summed_maps[i])
            path = f"{document_name}/highlighted_page_{i}.png"
            fig.savefig(path)
            plt.close(fig)
            highlighted_page_paths.append(path)

        return (highlighted_page_paths, relevancy_scores)

def get_model_and_processor():
    global model_singleton
    global processor_singleton
    global device_singleton
    try:
        return (model_singleton, processor_singleton, device_singleton)
    except NameError as e:
        print("Performing singleton initialization for model objects")
        model_name = "vidore/colqwen2-v1.0"
        device_singleton = get_torch_device("auto")
        model_singleton = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_singleton
        ).eval()
        processor_singleton = ColQwen2Processor.from_pretrained(model_name)
        return (model_singleton, processor_singleton, device_singleton)
