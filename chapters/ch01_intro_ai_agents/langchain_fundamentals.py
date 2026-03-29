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
    Base OpenAI Call
    """)
    return


@app.cell
def _():
    import getpass
    from openai import OpenAI

    return OpenAI, getpass


@app.cell
def _(getpass):
    OPENAI_API_KEY = getpass.getpass('Enter your OPENAI_API_KEY')
    return (OPENAI_API_KEY,)


@app.cell
def _(OPENAI_API_KEY, OpenAI):
    client = OpenAI(
        api_key = OPENAI_API_KEY
    )

    completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        { "role": "system", 
          "content": "You are a helpful AI assistant." },
        { "role": "user", 
          "content": "How many Greek temples are there in Paestum?" }
      ],
      temperature=0.7
    )
    return (completion,)


@app.cell
def _(completion):
    print(completion.choices[0].message.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    LangChain: Sentence completion example
    """)
    return


@app.cell
def _():
    from langchain_openai import ChatOpenAI
    import textwrap

    return ChatOpenAI, textwrap


@app.cell
def _(ChatOpenAI, OPENAI_API_KEY, mo, textwrap):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                     model_name="gpt-5-nano")
    _reply = llm.invoke("It's a hot day, I would like to go to the...")
    _raw_reply = _reply.content if hasattr(_reply, "content") else _reply
    _wrapped_reply = "\n".join(textwrap.wrap(str(_raw_reply), width=80))
    mo.Html(f"{_wrapped_reply}")
    return (llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Simple prompt
    """)
    return


@app.cell
def _(llm, mo):
    prompt_input = """Write a short message to remind users to be 
    vigilant about phishing attacks."""
    response = llm.invoke(prompt_input)

    mo.Html(f"{response.content}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Prompt instantiated thorugh a PromptTemplate
    """)
    return


@app.cell
def _(llm, mo):
    def _():
        from langchain_core.prompts import PromptTemplate

        segovia_aqueduct_text = """The Aqueduct of Segovia 
        (Spanish: Acueducto de Segovia) is a Roman aqueduct in Segovia, 
        Spain. It was built around the first century AD to channel water 
        from springs in the mountains 17 kilometres (11 mi) away to the 
        city's fountains, public baths and private houses, and was in 
        use until 1973. Its elevated section, with its complete arcade 
        of 167 arches, is one of the best-preserved Roman aqueduct 
        bridges and the foremost symbol of Segovia, as evidenced by 
        its presence on the city's coat of arms. The Old Town of 
        Segovia and the aqueduct, were declared a UNESCO World 
        Heritage Site in 1985. As the aqueduct lacks a legible 
        inscription (one was apparently located in the structure's 
        attic, or top portion[citation needed]), the date of 
        construction cannot be definitively determined. The general 
        date of the Aqueduct's construction was long a mystery, 
        although it was thought to have been during the 1st century AD, 
        during the reigns of the Emperors Domitian, Nerva, and Trajan. 
        At the end of the 20th century, Géza Alföldy deciphered the 
        text on the dedication plaque by studying the anchors that held 
        the now missing bronze letters in place. He determined that Emperor 
        Domitian (AD 81–96) ordered its construction[1] and the year 98 AD
        was proposed as the most likely date of completion.[2] However, 
        in 2016 archeological evidence was published which points to a 
        slightly later date, after 112 AD, during the government of 
        Trajan or in the beginning of the government of emperor Hadrian, 
        from 117 AD."""

        prompt_template = PromptTemplate.from_template("""You are an 
        experienced copywriter. Write a {num_words} words summary of 
        the following text, using a {tone} tone: {text}""")

        prompt_input = prompt_template.format(
            text=segovia_aqueduct_text, 
            num_words=20, 
            tone="knowledgeable and engaging")
        response = llm.invoke(prompt_input)
        return mo.Html(f"{response.content}")


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Chain example
    """)
    return


@app.cell
def _(ChatOpenAI, OPENAI_API_KEY, mo):
    def _():
        from langchain_core.prompts import PromptTemplate

        segovia_aqueduct_text = """The Aqueduct of Segovia 
        (Spanish: Acueducto de Segovia) is a Roman aqueduct in Segovia, 
        Spain. It was built around the first century AD to channel water 
        from springs in the mountains 17 kilometres (11 mi) away to the 
        city's fountains, public baths and private houses, and was in 
        use until 1973. Its elevated section, with its complete arcade 
        of 167 arches, is one of the best-preserved Roman aqueduct 
        bridges and the foremost symbol of Segovia, as evidenced by 
        its presence on the city's coat of arms. The Old Town of 
        Segovia and the aqueduct, were declared a UNESCO World 
        Heritage Site in 1985. As the aqueduct lacks a legible 
        inscription (one was apparently located in the structure's 
        attic, or top portion[citation needed]), the date of 
        construction cannot be definitively determined. The general 
        date of the Aqueduct's construction was long a mystery, 
        although it was thought to have been during the 1st century AD, 
        during the reigns of the Emperors Domitian, Nerva, and Trajan. 
        At the end of the 20th century, Géza Alföldy deciphered the 
        text on the dedication plaque by studying the anchors that held 
        the now missing bronze letters in place. He determined that Emperor 
        Domitian (AD 81–96) ordered its construction[1] and the year 98 AD
        was proposed as the most likely date of completion.[2] However, 
        in 2016 archeological evidence was published which points to a 
        slightly later date, after 112 AD, during the government of 
        Trajan or in the beginning of the government of emperor Hadrian, 
        from 117 AD."""
    
        prompt_template = PromptTemplate.from_template("You are an experienced copywriter. Write a {num_words} words summary of the following text, using a {tone} tone: {text}")
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                         model_name="gpt-5-nano")

        chain = prompt_template | llm
        response = chain.invoke({"text": segovia_aqueduct_text, 
                  "num_words": 20, 
                  "tone": "knowledgeable and engaging"})
        return mo.Html(f"{response.content}")


    _()
    return


if __name__ == "__main__":
    app.run()
