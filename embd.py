import os

from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity


client = genai.Client(api_key=os.environ.get("apikey_embd"))

question = "What is there to do in the historic part of Sacramento?"

f = open("info.txt")
info = f.read()
newinfo = info.split("\n\n")
f.close()
print(len(max(newinfo, key=len)))

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=newinfo
) 

def find(question: str) -> str:
    """ search on the notes and returns the most relevant fragment """
    q_emb = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
    )
    q_vec = q_emb.embeddings[0].values
    vectors = [e.values for e in result.embeddings]
    scores = cosine_similarity(vectors, [q_vec])
    top = scores.flatten().argsort()[-3:][::-1]

    return "\n\n".join(newinfo[i] for i in top)

conversation = [types.Content(role="user", parts=[types.Part(text=question)])]

for loop in range(5):
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=conversation,
        config=types.GenerateContentConfig(
            system_instruction=("Answer using only the information returned by the search tool. Do not add anything from your own knowledge. After at most three searches, answer with whatever you have. If the information is not there, say so instead of making it up."),
            tools=[find],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    if not response.function_calls:
        print(response.text)
        break

    conversation.append(response.candidates[0].content)

    for call in response.function_calls:
        print("buscando:", call.args["question"])
        tool_result = find(call.args["question"])
        conversation.append(
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=call.name,
                    response={"result": tool_result}
                )],
            )
        )
