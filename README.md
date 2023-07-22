# Artificial Intelligence for Learning Material Synthesis Processes of Thermoelectric Materials
Various calculation-based and data-driven methods have been proposed to discover high-performance thermoelectric materials for sustainable energy resources.
However, although several data-driven methods successfully discovered the chemical compositions of promising thermoelectric materials, the practical potential of the existing methods is still limited because there is a complex engineering problem between the discovered materials and the real-world material synthesis.
To tackle the engineering problem in material synthesis, we propose a multimodal graph-to-sequence model that predicts necessary synthesis operations and their engineering conditions from chemical compositions of precursor and desired product materials.
For an experimental evaluation, we constructed a benchmark dataset containing precursor materials, product materials, and synthesis processes of 771 unique thermoelectric materials.
The proposed method achieved prediction accuracy greater than 0.85 in Jaccard similarity and R2-score on the benchmark dataset.
Furthermore, the proposed method successfully generated material synthesis recipes described in the human language via large language models.

Reference: https://???.???

# Run
- ``exec_operations.py``: Train and evaluate a encoder-decoder model to predict the synthesis operations of the material synthesis recipes.
- ``exec_eng_conditions.py``: Train and evaluate XGBoost-based prediction models to predict the engineering conditions based on the generated sequence embeddings.
- ``exec_gen_recipe.py``: Generate a paragraph descring the material synthesis process based on the predicted synthesis operations and engineering conditions.

# Notes
- The ``save`` folder contains the pre-trained models and the experimental results in the paper.
- An API key of OpenAI is required to execute ``exec_gen_recipe.py``. Please visit [OpenAI_API](https://platform.openai.com/docs/api-reference) to get an API key of ChatGPT.
