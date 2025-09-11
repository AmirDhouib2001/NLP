import openai
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import List
import csv
from io import StringIO

load_dotenv()

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("API_KEY"),
)

# Prix des modèles (par million de tokens)
MODEL_PRICES = {
    "openai/gpt-oss-20b": {
        "input": 0.10,
        "output": 0.50,
        "speed": 1000  # tokens/sec
    },
    "llama-3.3-70b-versatile": {
        "input": 0.59,
        "output": 0.79,
        "speed": 394  # tokens/sec
    }
}

def calculate_cost_and_time(usage, model_name):
    """Calcule le coût et temps basé sur l'usage des tokens"""
    if model_name not in MODEL_PRICES:
        return None, None

    prices = MODEL_PRICES[model_name]
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    # Calcul du coût (prix par million de tokens)
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    # Calcul du temps théorique (basé sur les tokens de sortie)
    time_seconds = output_tokens / prices["speed"]

    return total_cost, time_seconds

# Call classic
# reply = client.chat.completions.create(
#     messages=[{"role": "user", "content": "Who is the best NLP teacher?"}],
#     model="llama-3.3-70b-versatile",
# )
# print(reply.choices[0].message.content)


# Call with structured output
class ComicName(BaseModel):
    name: str

class ComicNames(BaseModel):
    names: List[str]

# reply = client.chat.completions.parse(
#     messages=[{"role": "user", "content": "Extract the comedian's name in this video title: 'Ne me parlez plus d IA - la chronique de Thomas VDB'"}],
#     model="openai/gpt-oss-20b",
#     response_format=ComicName,
# )

# print(reply.choices[0].message.content)


def extract_comic_names(video_name: str) -> List[str]:
    try:
        reply = client.chat.completions.parse(
            messages=[{
                "role": "user",
                "content": f"Extract the comedian's name in this video title: '{video_name}'"
            }],
            model="openai/gpt-oss-20b",
            response_format=ComicNames,
        )

        # Calculer coût et temps
        cost, time_sec = calculate_cost_and_time(reply.usage, "openai/gpt-oss-20b")
        print(f"Usage - Input: {reply.usage.prompt_tokens}, Output: {reply.usage.completion_tokens}")
        print(f"Coût: ${cost:.6f}, Temps: {time_sec:.3f}s")

        parsed_response = reply.choices[0].message.parsed
        return parsed_response.names if parsed_response else []

    except Exception as e:
        print(f"Erreur lors de l'extraction: {e}")
        return []


def extract_comic_names_batch(video_names: List[str]) -> List[str]:

    try:
        # Créer la liste des vidéos pour le prompt
        video_list = "\n".join([f"- {video}" for video in video_names])

        reply = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"""Extract comedian names from these France Inter video titles.

Video titles:
{video_list}

Start your reply with ```csv
video_name;comic_names"""
            }],
            model="llama-3.3-70b-versatile",
        )

        response_content = reply.choices[0].message.content

        # Calculer coût et temps
        cost, time_sec = calculate_cost_and_time(reply.usage, "llama-3.3-70b-versatile")
        print(f"Usage - Input tokens: {reply.usage.prompt_tokens}, Output tokens: {reply.usage.completion_tokens}")
        print(f"Cost: ${cost:.6f}")

        # print("response_content: ", response_content)

        # Parser la réponse CSV
        comic_names = []
        if "```csv" in response_content:
            # Extraire le contenu CSV
            csv_start = response_content.find("```csv") + 6
            csv_end = response_content.find("```", csv_start)
            if csv_end == -1:
                csv_content = response_content[csv_start:].strip()
            else:
                csv_content = response_content[csv_start:csv_end].strip()

            # Parser le CSV
            csv_reader = csv.reader(StringIO(csv_content), delimiter=';')
            next(csv_reader, None)

            for row in csv_reader:
                if len(row) >= 2 and row[1].strip():
                    names = [name.strip() for name in row[1].split(',')]
                    comic_names.extend([name for name in names if name])

        return comic_names

    except Exception as e:
        print(f"Erreur lors de l'extraction batch: {e}")
        return []


if __name__ == "__main__":
    test_titles = [
        "Ne me parlez plus d IA - la chronique de Thomas VDB",
        "Le Barbecue Disney - La chanson de Frédéric Fromet",
        "Pickles de navet express - Les recettes de François-Régis Gaudry",
        "Ras-Le-Bol des armes que Stallone fasse des pichenettes - Tanguy Pastureau maltraite l'info",
        "La Belle et les Bêtes - Gérémy Crédeville part en live",
        "Le jouranl info de test pour llm."
    ]

    # # Test fonction individuelle
    # for title in test_titles:
    #     comic_names = extract_comic_names(title)
    #     print(f"Titre: {title}")
    #     print(f"Comiques trouvés: {comic_names}")
    #     print()

    print("="*50)
    print("Test fonction batch:")

    # Test fonction batch
    batch_comic_names = extract_comic_names_batch(test_titles)
    print(f"Tous les comiques trouvés: {batch_comic_names}")
