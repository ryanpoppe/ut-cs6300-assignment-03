import pandas as pd
import random
import json
from transformers import pipeline
from smolagents import OpenAIServerModel, ToolCallingAgent

# --------------------------
# CONFIG
# --------------------------
DATASET_PATH = "wiki_movie_plots_deduped.csv"  # downloaded from Kaggle
N_QUERIES = 500  # total queries to generate
OUTPUT_FILE = "queries.json"

# --------------------------
# STEP 1: Load dataset
# --------------------------
df = pd.read_csv(DATASET_PATH)
plots = df["Plot"].dropna().tolist()

# safety measure: shuffle so we don’t always use the first movies
random.shuffle(plots)

# --------------------------
# STEP 2: Summarization pipeline
# --------------------------
print("Loading summarization model...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)


def summarize_plot(plot, max_len=100):
    """Summarize a long movie plot into a shorter synopsis."""
    try:
        summary = summarizer(
            plot,
            max_length=max_len,
            min_length=30,
            truncation=True,
            do_sample=False,
        )
        return summary[0]["summary_text"]
    except Exception as e:
        print("Summarization failed:", e)
        return None


# --------------------------
# STEP 3: Query generation (LLM)
# --------------------------

# Configure the model to use LM Studio's local API endpoint
# LM Studio typically runs on http://localhost:1234/v1
model = OpenAIServerModel(
    model_id="local-model",  # This can be any name, LM Studio will use whatever model you have loaded
    api_base="http://localhost:1234/v1",  # Default LM Studio API endpoint
    api_key="not-needed",  # LM Studio doesn't require an API key by default
)

# Create a simple agent using the local model
agent = ToolCallingAgent(
    name="LocalLLMAgent",
    model=model,
    tools=[],  # Empty list of tools
)


def generate_query(summary):
    """Generate a natural-language search query from a movie summary."""
    prompt = (
        f"Movie summary: {summary}\n"
        f"Generate a single natural-language search query a user might ask to find other movies like this one. Limit the query to 20 words or less. Respond with just the query, no other text, in the form of 'Find movies similar to ...'."
    )
    response = agent.run(prompt)
    print(f"LLM response: {response}")
    return response.strip()


# --------------------------
# STEP 4: Run on dataset
# --------------------------
queries = list()
for plot in plots:
    print(f"\nProcessing plot {len(queries)+1}/{N_QUERIES}...")
    if len(queries) >= N_QUERIES:
        break

    summary = summarize_plot(plot)
    if not summary:
        continue

    try:
        qs = generate_query(summary)
        queries.append(qs)
    except Exception as e:
        print("Query generation failed:", e)
        continue

print(f"Generated {len(queries)} queries.")

# --------------------------
# STEP 5: Save
# --------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(queries[:N_QUERIES], f, indent=2)

print(f"Saved queries to {OUTPUT_FILE}")
