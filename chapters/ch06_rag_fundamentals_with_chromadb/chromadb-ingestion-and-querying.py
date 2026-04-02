import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Chroma DB ingestion and Q&A
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Ingestion
    """)
    return


@app.cell
def _():
    import chromadb
    chroma_client = chromadb.Client()
    return (chroma_client,)


@app.cell
def _(chroma_client):
    tourism_collection = chroma_client.create_collection(
        name="tourism_collection")
    return (tourism_collection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### INSERTING THE CONTENT
    """)
    return


@app.cell
def _(tourism_collection):
    tourism_collection.add(
        documents=[
            "Paestum, Greek Poseidonia, ancient city in southern Italy near the west coast, 22 miles (35 km) southeast of modern Salerno and 5 miles (8 km) south of the Sele (ancient Silarus) River. Paestum is noted for its splendidly preserved Greek temples.", 
            "Poseidonia was probably founded about 600 BC by Greek colonists from Sybaris, along the Gulf of Taranto, and it had become a flourishing town by 540, judging from its temples. After many years’ resistance the city came under the domination of the Lucanians (an indigenous Italic people) sometime before 400 BC, after which its name was changed to Paestum. Alexander, the king of Epirus, defeated the Lucanians at Paestum about 332 BC, but the city remained Lucanian until 273, when it came under Roman rule and a Latin colony was founded there. The city supported Rome during the Second Punic War. The locality was still prosperous during the early years of the Roman Empire, but the gradual silting up of the mouth of the Silarus River eventually created a malarial swamp, and Paestum was finally deserted after being sacked by Muslim raiders in AD 871. The abandoned site’s remains were rediscovered in the 18th century.",
            "The ancient Greek part of Paestum consists of two sacred areas containing three Doric temples in a remarkable state of preservation. During the ensuing Roman period a typical forum and town layout grew up between the two ancient Greek sanctuaries. Of the three temples, the Temple of Athena (the so-called Temple of Ceres) and the Temple of Hera I (the so-called Basilica) date from the 6th century BC, while the Temple of Hera II (the so-called Temple of Neptune) was probably built about 460 BC and is the best preserved of the three. The Temple of Peace in the forum is a Corinthian-Doric building begun perhaps in the 2nd century BC. Traces of a Roman amphitheatre and other buildings, as well as intersecting main streets, have also been found. The circuit of the town walls, which are built of travertine blocks and are 15–20 feet (5–6 m) thick, is about 3 miles (5 km) in circumference. In July 1969 a farmer uncovered an ancient Lucanian tomb that contained Greek frescoes painted in the early classical style. Paestum’s archaeological museum contains these and other treasures from the site."
        ],
        metadatas=[
            {"source": "https://www.britannica.com/place/Paestum"}, 
            {"source": "https://www.britannica.com/place/Paestum"},
            {"source": "https://www.britannica.com/place/Paestum"}
        ],
        ids=["paestum-br-01", "paestum-br-02", "paestum-br-03"]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### PERFORMING A SEMANTIC SEARCH
    """)
    return


@app.cell
def _(tourism_collection):
    results = tourism_collection.query(
        query_texts=["How many Doric temples are in Paestum"],
        n_results=1
    )
    print(results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RAG from scratch
    """)
    return


@app.cell
def _():
    from openai import OpenAI
    import getpass

    return OpenAI, getpass


@app.cell
def _(getpass):
    OPENAI_API_KEY = getpass.getpass('Enter your OPENAI_API_KEY')
    return (OPENAI_API_KEY,)


@app.cell
def _(OPENAI_API_KEY, OpenAI):
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return (openai_client,)


@app.cell
def _(tourism_collection):
    def query_vector_database(question):
        results = tourism_collection.query(
            query_texts=[question],
            n_results=1
        )
        results_text = results['documents'][0][0]
        return results_text

    return (query_vector_database,)


@app.cell
def _(query_vector_database):
    results_text = query_vector_database("How many Doric temples are in Paestum")
    print(results_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Invoking the LLM
    """)
    return


@app.function
def prompt_template(question, text):
    return f'Use the following pieces of retrieved context to➥answer the question. Only use the retrieved context to answer the question. If you don\'t know the answer, or the answer is not contained in the retrieved context,just say that you don\'t know. Use three sentences maximum and keep the answer concise. \nQuestion: {question}\nContext: {text}. Remember: if you do not know, just say: I do not know. Do not make up an➥answer. For example do not say the three temples have got a total of three columns. \nAnswer:'


@app.cell
def _(openai_client):
    def execute_llm_prompt(prompt_input):
        prompt_response = openai_client.chat.completions.create(
            model='gpt-5-nano',
            messages=[
                {"role": "system", "content": "You are an assistant for question-answering tasks."},
                {"role": "user", "content": prompt_input}
            ])
        return prompt_response

    return (execute_llm_prompt,)


@app.cell
def _(execute_llm_prompt, query_vector_database):
    trick_question = "How many columns have the three temples got in total?"
    tq_result_text = query_vector_database(trick_question)
    tq_prompt = prompt_template(trick_question, tq_result_text)
    tq_prompt_response = execute_llm_prompt(tq_prompt)
    print(tq_prompt_response)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Building the chatbot
    """)
    return


@app.cell
def _(execute_llm_prompt, query_vector_database):
    def my_chatbot(question):
        results_text = query_vector_database(question)
        prompt_input = prompt_template(question, results_text)
        prompt_output = execute_llm_prompt(prompt_input)
        return prompt_output


    return (my_chatbot,)


@app.cell
def _(my_chatbot):
    question = """Let me know how many temples there
        are in Paestum, who constructed them, and what
        architectural style they are"""
    result = my_chatbot(question)
    print(result)
    return


if __name__ == "__main__":
    app.run()
