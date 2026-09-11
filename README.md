# sacramento-rag-agent

Small RAG agent I built to understand how retrieval and tool calling work, no framework.

It reads `info.txt` (notes about Sacramento), splits it by paragraph, embeds each
chunk with gemini-embedding-001 and finds the closest ones to the question with
cosine similarity. That search is wrapped as a tool called `find`. Then there's a
loop where Gemini decides if it needs to search, gets the top 3 chunks, and answers
only with that. If the info isn't there it has to say so.

I turned off automatic function calling so I'd have to write the loop myself.

To run it:

    pip install google-genai scikit-learn
    export apikey_embd=your_key
    python embd.py

You need an `info.txt` in the same folder, any text separated by blank lines.

Things I'd do next: cache the embeddings (right now they're recomputed every run),
add a second tool, write a couple of tests.
